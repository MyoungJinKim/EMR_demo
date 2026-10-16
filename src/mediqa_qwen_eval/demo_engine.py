"""Lazy, offline speech and EMR inference. The HTTP server owns the inference lock."""
from __future__ import annotations

import gc
import json
import os
import platform
import re
import struct
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .paths import PROJECT_ROOT
from .postprocess import postprocess_note
from .soap import soap_messages, soap_translation_messages, normalize_soap_note, validate_source_numbers, place_explicit_clinician_statements
from .demo_stt import MlxWhisper, MLX_STT_MODEL, KOREAN_VOCABULARY, is_mlx_whisper_model, speech_chunks

DEFAULT_EMR_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-9B-4bit"
MODEL_REVISIONS = {
    DEFAULT_EMR_MODEL: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    DEFAULT_MLX_MODEL: "8b2b98c00a6b4d291155e4890773ca8f769aee53",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mlx-community/Qwen2.5-7B-Instruct-4bit": "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}
CACHE_DIR = PROJECT_ROOT / ".demo-cache"
MAX_AUDIO_SECONDS = 600
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 2048
PREPARE_HELP = "python scripts/prepare_voice_demo.py를 실행한 뒤 다시 시도하세요."
CONTROL_TOKENS = re.compile(r"<\|[^<>]*\|>|</?s>|\[/?INST\]", re.IGNORECASE)
HAN_CHARACTERS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
DAY_WORDS = {"하루": 1, "이틀": 2, "사흘": 3, "나흘": 4, "닷새": 5, "엿새": 6, "이레": 7, "여드레": 8, "아흐레": 9, "열흘": 10, "보름": 15}
# shortcut: normalize only explicit duration contexts; extend if other counters are mistranslated.
DAY_DURATION = re.compile(
    r"(?<!\w)(" + "|".join(DAY_WORDS) + r")"
    r"(?=\s*(?:전(?:부터|까지)?|후(?:부터|까지|에도|에)?|동안|째|이상|정도|간|만에)(?:\s|[.,!?]|$))"
)


class InvalidModelError(ValueError):
    """A model preparation error safe to display in the demo."""


class InvalidAudioError(ValueError):
    """An audio validation error safe to display in the demo."""


def validate_safetensors(model_path: Path) -> None:
    """Check shard existence and header lengths without reading gigabytes of tensors."""
    try:
        indexes = list(model_path.glob("*.safetensors.index.json"))
        if indexes:
            names = set(json.loads(indexes[0].read_text())["weight_map"].values())
            files = [model_path / name for name in names]
        else:
            files = list(model_path.glob("*.safetensors"))
        if not files:
            raise ValueError("missing weights")
        for path in files:
            if path.parent != model_path or not path.is_file():
                raise ValueError("missing shard")
            with path.open("rb") as handle:
                size_bytes = handle.read(8)
                if len(size_bytes) != 8:
                    raise ValueError("missing header")
                header_size = struct.unpack("<Q", size_bytes)[0]
                if not 0 < header_size <= min(path.stat().st_size - 8, 100_000_000):
                    raise ValueError("invalid header")
                header = json.loads(handle.read(header_size))
            offsets = [item["data_offsets"][-1] for key, item in header.items() if key != "__metadata__"]
            if path.stat().st_size != 8 + header_size + max(offsets):
                raise ValueError("incomplete weights")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, struct.error) as error:
        raise InvalidModelError(
            "EMR 모델 파일이 없거나 다운로드가 불완전합니다. 기존 인증 모델은 덮어쓰지 말고 "
            "완전한 모델 폴더를 EMR_MODEL_PATH로 지정하세요. " + PREPARE_HELP
        ) from error


def decode_audio_bounded(path: Path) -> tuple[Any, float]:
    """Bound decoding itself, including containers without duration metadata."""
    import av
    import numpy as np

    chunks = []
    sample_count = 0
    decoded_seconds = 0.0
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise InvalidAudioError("오디오 트랙이 없습니다. 음성 파일을 선택하세요.")
            if container.duration and container.duration / av.time_base > MAX_AUDIO_SECONDS:
                raise InvalidAudioError("음성은 10분 이하로 녹음하거나 업로드하세요.")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                if not frame.sample_rate:
                    raise InvalidAudioError("오디오의 샘플레이트를 확인할 수 없습니다.")
                decoded_seconds += frame.samples / frame.sample_rate
                if decoded_seconds > MAX_AUDIO_SECONDS + 1 / 16000:
                    raise InvalidAudioError("음성은 10분 이하로 녹음하거나 업로드하세요.")
                frame.pts = None
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().flatten()
                    sample_count += samples.size
                    if sample_count > MAX_AUDIO_SECONDS * 16000:
                        raise InvalidAudioError("음성은 10분 이하로 녹음하거나 업로드하세요.")
                    chunks.append(samples)
            for converted in resampler.resample(None):
                samples = converted.to_ndarray().flatten()
                sample_count += samples.size
                if sample_count > MAX_AUDIO_SECONDS * 16000:
                    raise InvalidAudioError("음성은 10분 이하로 녹음하거나 업로드하세요.")
                chunks.append(samples)
    except InvalidAudioError:
        raise
    except Exception as error:
        raise ValueError("음성 파일을 읽을 수 없습니다. WAV, MP3, M4A 또는 WebM 파일을 사용하세요.") from error
    if not sample_count:
        raise ValueError("음성이 비어 있습니다. 다시 녹음하세요.")
    waveform = np.concatenate(chunks).astype(np.float32) / 32768.0
    if not np.any(waveform):
        raise ValueError("소리가 감지되지 않았습니다. 마이크 입력을 확인하세요.")
    return waveform, sample_count / 16000.0


def note_messages(text: str, language: str) -> list[dict[str, str]]:
    if language not in {"ko", "en"}:
        raise ValueError("노트 언어는 한국어 또는 영어를 선택하세요.")
    if not text.strip():
        raise ValueError("전사 텍스트가 없습니다. 먼저 녹음하거나 텍스트를 입력하세요.")
    if len(text) > 20000:
        raise ValueError("전사 텍스트는 20,000자 이하로 입력하세요.")
    transcript = CONTROL_TOKENS.sub(" ", text).strip()
    if not transcript:
        raise ValueError("임상 내용을 포함한 전사 텍스트를 입력하세요.")
    transcript = DAY_DURATION.sub(lambda match: f"{DAY_WORDS[match.group(1)]}일", transcript)
    return soap_messages(transcript, language)


class DemoEngine:
    def __init__(self) -> None:
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.stt_name = os.getenv("STT_MODEL", MLX_STT_MODEL if apple_silicon else "large-v3")
        stt_path = Path(self.stt_name).expanduser()
        self.stt_backend = "mlx" if self.stt_name.startswith("mlx-community/whisper-") or is_mlx_whisper_model(stt_path) else "faster-whisper"
        self.stt_device = os.getenv("STT_DEVICE", "metal" if self.stt_backend == "mlx" else "cpu")
        default_model = DEFAULT_MLX_MODEL if apple_silicon else DEFAULT_EMR_MODEL
        self.emr_name = os.getenv("EMR_MODEL", default_model)
        self.emr_path = os.getenv("EMR_MODEL_PATH", "")
        self.adapter_path = os.getenv("EMR_ADAPTER_PATH", "")
        self.device = os.getenv("EMR_DEVICE", "auto")
        self.base_url = os.getenv("EMR_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("EMR_API_KEY", "")
        if self.stt_device not in {"cpu", "cuda", "metal"} or self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("STT_DEVICE는 cpu/cuda/metal, EMR_DEVICE는 auto/cpu/mps/cuda를 사용하세요.")
        if self.stt_backend == "mlx" and (not apple_silicon or self.stt_device != "metal"):
            raise ValueError("MLX 음성 모델은 Apple Silicon Mac의 STT_DEVICE=metal로 실행하세요.")
        if self.stt_backend != "mlx" and self.stt_device == "metal":
            raise ValueError("Metal 음성 인식에는 MLX STT 모델을 지정하세요.")
        if self.base_url:
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("EMR_BASE_URL에 인증정보 없는 http(s) API 주소를 지정하세요.")
            if self.adapter_path:
                raise ValueError("원격 EMR 서버에서는 서버에 LoRA를 적용한 뒤 EMR_MODEL을 지정하세요.")
        self._stt = None
        self._model = None
        self._tokenizer = None
        self._remote_loaded = False
        self._stt_prepared = False
        self._emr_prepared = False
        self._backend = "mlx" if self.emr_name.startswith("mlx-community/") and not self.emr_path else "transformers"
        self._context_length = 32768
        self._korean_logit_bias = None

    def health(self) -> dict[str, Any]:
        return {
            "ready": self._stt_prepared and self._emr_prepared,
            "format": "SOAP",
            "stt": {"model": Path(self.stt_name).name, "loaded": self._stt is not None,
                    "backend": self.stt_backend, "device": self.stt_device},
            "emr": {
                "model": Path(self.emr_path).name if self.emr_path else self.emr_name,
                "loaded": self._remote_loaded if self.base_url else self._model is not None,
                "backend": "openai-compatible" if self.base_url else self._backend,
                "device": self.device, "adapter": bool(self.adapter_path),
            },
        }

    def _load_stt(self) -> None:
        if self._stt is not None:
            return
        try:
            if self.stt_backend == "mlx":
                # 16 GB Macs cannot comfortably retain both full Whisper and the EMR model.
                if self._model is not None:
                    self._model = self._tokenizer = None
                    gc.collect()
                    if self._backend == "transformers" and self.device == "mps":
                        import torch

                        torch.mps.empty_cache()
                    import mlx.core as mx

                    mx.clear_cache()
                path = Path(self.stt_name).expanduser()
                if not path.is_dir():
                    path = CACHE_DIR / self.stt_name.split("/")[-1]
                stt = MlxWhisper(path)
                stt.load()
                self._stt = stt
                self._stt_prepared = True
                return
            from faster_whisper import WhisperModel

            explicit_path = Path(self.stt_name).expanduser()
            source = str(explicit_path) if explicit_path.is_dir() else self.stt_name
            self._stt = WhisperModel(
                source, device=self.stt_device,
                compute_type="int8" if self.stt_device == "cpu" else "float16",
                download_root=str(CACHE_DIR / "whisper"), local_files_only=True,
            )
            self._stt_prepared = True
        except Exception as error:
            raise RuntimeError("STT 모델을 불러올 수 없습니다. 데모 의존성과 STT_MODEL 경로를 확인하세요. " + PREPARE_HELP) from error

    def _load_emr(self) -> None:
        if self._model is not None or self._remote_loaded:
            return
        if self.base_url:
            response = self._remote_request("models")
            if self.emr_name not in {item.get("id") for item in response.get("data", [])}:
                raise RuntimeError("EMR 서버에서 지정한 모델을 찾을 수 없습니다. EMR_MODEL을 확인하세요.")
            self._remote_loaded = True
            self._emr_prepared = True
            return
        try:
            if self.stt_backend == "mlx" and self._stt is not None:
                self._stt.unload()
                self._stt = None
            from huggingface_hub import snapshot_download

            local_path = CACHE_DIR / self.emr_name.split("/")[-1]
            if self.emr_path:
                model_path = Path(self.emr_path).expanduser().resolve()
            elif local_path.is_dir():
                model_path = local_path
            else:
                revision = MODEL_REVISIONS.get(self.emr_name, "main")
                model_path = Path(snapshot_download(self.emr_name, revision=revision, local_files_only=True))
            validate_safetensors(model_path)
            config = json.loads((model_path / "config.json").read_text())
            self._context_length = int(config.get("text_config", config).get("max_position_embeddings", 32768))
            if config.get("quantization"):
                if platform.system() != "Darwin" or platform.machine() != "arm64":
                    raise InvalidModelError("MLX 모델은 Apple Silicon Mac에서 사용하세요. 다른 장치에서는 일반 Qwen 모델 폴더를 EMR_MODEL_PATH로 지정하세요.")
                if self.adapter_path:
                    raise InvalidModelError("MLX 데모 모델에는 PEFT LoRA를 적용할 수 없습니다. 인증 LoRA는 일반 Qwen 모델과 Transformers로 실행하세요.")
                if self.device not in {"auto", "mps", "metal"}:
                    raise InvalidModelError("MLX 모델은 Mac GPU를 사용합니다. EMR_DEVICE=auto로 실행하세요.")
                from mlx_lm import load

                model, tokenizer = load(str(model_path), tokenizer_config={"local_files_only": True})
                self._model, self._tokenizer = model, tokenizer
                self._backend, self.device = "mlx", "metal"
                self._emr_prepared = True
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

            self._backend = "transformers"
            if self.adapter_path:
                adapter = Path(self.adapter_path).expanduser().resolve()
                validate_safetensors(adapter)
                if not (adapter / "adapter_config.json").is_file():
                    raise InvalidModelError("EMR_ADAPTER_PATH에 adapter_config.json이 없습니다.")
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float32 if self.device == "cpu" else torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model_class = AutoModelForImageTextToText if config.get("model_type") == "qwen3_5" else AutoModelForCausalLM
            model = model_class.from_pretrained(
                model_path, local_files_only=True, dtype=dtype,
                attn_implementation="sdpa", low_cpu_mem_usage=True,
            ).to(self.device)
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
            model.eval()
            self._tokenizer, self._model = tokenizer, model
            self._emr_prepared = True
        except InvalidModelError:
            raise
        except Exception as error:
            raise RuntimeError("EMR 모델을 불러올 수 없습니다. EMR_MODEL_PATH, 실행 장치와 사용 가능한 메모리를 확인하세요. " + PREPARE_HELP) from error

    def _remote_request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        try:
            request = Request(f"{self.base_url}/{path}", data=data, headers=headers)
            with urlopen(request, timeout=180) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("oversized response")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("invalid response")
            return result
        except HTTPError as error:
            raise RuntimeError(f"EMR 서버가 요청을 거절했습니다 (HTTP {error.code}). 서버의 모델 설정과 인증을 확인하세요.") from error
        except (URLError, OSError, ValueError) as error:
            raise RuntimeError("EMR 서버에 연결하거나 응답을 읽을 수 없습니다. 서버 주소와 실행 상태를 확인하세요.") from error

    def warmup(self) -> dict[str, Any]:
        self._load_stt()
        self._load_emr()
        return self.health()

    def transcribe(self, path: Path, language: str = "ko") -> dict[str, Any]:
        if language not in {"ko", "en", "auto"}:
            raise ValueError("음성 언어는 한국어, 영어 또는 자동 감지를 선택하세요.")
        started = time.perf_counter()
        waveform, duration = decode_audio_bounded(path)
        chunks = None
        if self.stt_backend == "mlx":
            try:
                chunks = speech_chunks(waveform)
            except Exception as error:
                raise RuntimeError("음성 구간을 분석하지 못했습니다. 짧고 선명한 녹음으로 다시 시도하세요.") from error
            if not chunks:
                raise ValueError("발화를 감지하지 못했습니다. 마이크와 녹음 언어를 확인하고 다시 녹음하세요.")
        self._load_stt()
        try:
            if self.stt_backend == "mlx":
                rows, detected = self._stt.transcribe(waveform, None if language == "auto" else language, chunks=chunks)
            else:
                segments, info = self._stt.transcribe(
                    waveform, language=None if language == "auto" else language,
                    beam_size=5, vad_filter=True, temperature=0,
                    condition_on_previous_text=False,
                    hotwords=KOREAN_VOCABULARY if language == "ko" else None,
                )
                rows = [{"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
                        for segment in segments if segment.text.strip()]
                detected = info.language
        except Exception as error:
            raise RuntimeError("음성 전사에 실패했습니다. 짧고 선명한 녹음으로 다시 시도하세요.") from error
        text = " ".join(row["text"] for row in rows)
        if not text:
            raise ValueError("발화를 감지하지 못했습니다. 마이크와 녹음 언어를 확인하고 다시 녹음하세요.")
        return {"text": text, "segments": rows, "language": detected,
                "duration_seconds": duration, "elapsed_seconds": time.perf_counter() - started,
                "model": self.health()["stt"]["model"]}

    def generate(self, text: str, language: str = "ko") -> dict[str, Any]:
        result = self._generate(note_messages(text, language), language)
        if language == "en":
            result["note"] = place_explicit_clinician_statements(result["note"])
        return result

    def translate_to_korean(self, english_note: str, source_text: str = "") -> dict[str, Any]:
        transcript = CONTROL_TOKENS.sub(" ", source_text).strip()
        result = self._generate(soap_translation_messages(english_note, transcript), "ko", repair_language=True)
        try:
            validate_source_numbers(english_note, result["note"])
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        return result

    def _generate(self, messages: list[dict[str, str]], language: str, *, repair_language: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        self._load_emr()
        if self.base_url:
            response = self._remote_request("chat/completions", {
                "model": self.emr_name, "messages": messages,
                "temperature": 0, "max_tokens": MAX_NEW_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            try:
                choice = response["choices"][0]
                raw_output = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "stop")
                if not isinstance(raw_output, str):
                    raise ValueError("missing text")
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise RuntimeError("EMR 서버에서 노트 텍스트를 반환하지 않았습니다.") from error
        elif self._backend == "mlx":
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > min(MAX_INPUT_TOKENS, self._context_length - MAX_NEW_TOKENS):
                raise ValueError("대화가 모델 입력 한도를 초과합니다. 임상 내용이 손실되지 않도록 대화를 나눠 생성하세요.")
            if language == "ko" and self._korean_logit_bias is None:
                # Partial UTF-8 tokens also form valid Korean; the final language check handles assembled Han.
                self._korean_logit_bias = {
                    token_id: -float("inf") for token_id in self._tokenizer.get_vocab().values()
                    if (decoded := self._tokenizer.decode([token_id]))
                    and HAN_CHARACTERS.search(decoded)
                }
            parts, finish_reason = [], "stop"
            try:
                for response in stream_generate(
                    self._model, self._tokenizer, prompt=input_ids,
                    max_tokens=MAX_NEW_TOKENS, sampler=make_sampler(temp=0),
                    logits_processors=make_logits_processors(
                        logit_bias=self._korean_logit_bias if language == "ko" else None,
                    ),
                ):
                    parts.append(response.text)
                    if response.finish_reason:
                        finish_reason = response.finish_reason
            except Exception as error:
                raise RuntimeError("EMR 생성에 실패했습니다. 입력 길이와 사용 가능한 메모리를 확인하세요.") from error
            raw_output = "".join(parts)
        else:
            import torch

            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_length = encoded["input_ids"].shape[1]
            context_length = int(getattr(self._model.config, "max_position_embeddings", 32768))
            if input_length > min(MAX_INPUT_TOKENS, context_length - MAX_NEW_TOKENS):
                raise ValueError("대화가 모델 입력 한도를 초과합니다. 임상 내용이 손실되지 않도록 대화를 나눠 생성하세요.")
            device = self._model.get_input_embeddings().weight.device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            try:
                with torch.inference_mode():
                    generated = self._model.generate(
                        **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        repetition_penalty=1.05, use_cache=True,
                        pad_token_id=self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
                output_ids = generated[0, input_length:]
                raw_output = self._tokenizer.decode(output_ids, skip_special_tokens=True)
                finish_reason = "length" if len(output_ids) >= MAX_NEW_TOKENS and int(output_ids[-1]) != self._tokenizer.eos_token_id else "stop"
            except Exception as error:
                raise RuntimeError("EMR 생성에 실패했습니다. 입력 길이와 사용 가능한 메모리를 확인하세요.") from error
        note = postprocess_note(raw_output, trim_repetition=False)
        if not note:
            raise RuntimeError("EMR 모델이 빈 노트를 반환했습니다. 전사 내용을 확인하고 다시 생성하세요.")
        if finish_reason == "length":
            raise RuntimeError("SOAP 출력이 길이 한도로 중단되었습니다. 대화를 나눠 다시 생성하세요.")
        if "<think>" in note or "</think>" in note:
            raise RuntimeError("모델이 SOAP 노트 대신 추론 과정을 반환했습니다. 모델의 thinking 설정을 확인하세요.")
        if language == "ko" and (HAN_CHARACTERS.search(note) or not re.search(r"[가-힣]", note)):
            if repair_language:
                corrected = self._generate(messages + [
                    {"role": "assistant", "content": note},
                    {"role": "user", "content": (
                        "위 번역에 한국어가 아닌 표현이 섞였습니다. 영어 원문과 대조하여 모든 문장을 자연스러운 "
                        "한국어로 교정하세요. 중국어와 한자를 한국어로 옮기고, 약물명·수치·단위·부정·미확인 "
                        "상태·시점과 SOAP 섹션 소속은 보존하세요. 사실을 추가하거나 생략하지 마세요. "
                        "설명 없이 교정한 한국어 SOAP 전체만 출력하세요."
                    )},
                ], "ko")
                corrected["elapsed_seconds"] = time.perf_counter() - started
                return corrected
            raise RuntimeError("한국어 노트가 요청한 언어로 생성되지 않아 결과를 표시하지 않았습니다. 전사문을 확인하고 다시 생성하세요.")
        try:
            note = normalize_soap_note(note, language)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        return {"note": note, "elapsed_seconds": time.perf_counter() - started,
                "model": self.health()["emr"]["model"], "format": "SOAP", "finish_reason": finish_reason, "warnings": []}
