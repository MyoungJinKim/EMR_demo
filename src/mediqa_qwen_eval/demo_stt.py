"""Offline MLX Whisper with voice detection and bounded decoding windows."""
from __future__ import annotations

import gc
import json
from importlib import import_module
from pathlib import Path
from typing import Any

MLX_STT_MODEL = "mlx-community/whisper-large-v3-turbo"
MLX_STT_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
MLX_STT_SHA256 = "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6"
KOREAN_VOCABULARY = "복용, 처방, 알레르기, 활력징후, 과거력, 현병력, 통증, 증상, 검사, 진료, 혈뇨, 임의로, 확인한 뒤"
SAMPLE_RATE = 16000


def is_mlx_whisper_model(path: Path) -> bool:
    try:
        config = json.loads((path / "config.json").read_text())
        dimensions = (
            "n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
            "n_vocab", "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer",
        )
        return (
            isinstance(config, dict)
            and config.get("model_type", "whisper") == "whisper"
            and all(type(config.get(key)) is int and config[key] > 0 for key in dimensions)
            and any((path / name).is_file() for name in ("weights.npz", "weights.safetensors"))
        )
    except (OSError, ValueError):
        return False


def speech_chunks(waveform: Any) -> list[dict[str, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # Keep a pause around each utterance; silence must not be decoded into invented speech.
    return get_speech_timestamps(
        waveform,
        VadOptions(max_speech_duration_s=28, min_silence_duration_ms=2000, speech_pad_ms=400),
        sampling_rate=SAMPLE_RATE,
    )


class MlxWhisper:
    def __init__(self, model_path: Path) -> None:
        self.model_path = str(model_path.resolve())
        self._module: Any = None

    def load(self) -> None:
        if self._module is not None:
            return
        path = Path(self.model_path)
        if not is_mlx_whisper_model(path):
            raise ValueError("STT 모델 파일이 없습니다. python scripts/prepare_voice_demo.py를 실행하세요.")
        import mlx.core as mx

        module = import_module("mlx_whisper.transcribe")
        module.ModelHolder.get_model(self.model_path, mx.float16)
        self._module = module

    def unload(self) -> None:
        if self._module is None:
            return
        holder = self._module.ModelHolder
        if holder.model_path == self.model_path:
            holder.model = None
            holder.model_path = None
        self._module = None
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()

    def transcribe(self, waveform: Any, language: str | None = "ko", *,
                   chunks: list[dict[str, int]] | None = None) -> tuple[list[dict[str, Any]], str]:
        if chunks is None:
            chunks = speech_chunks(waveform)
        if not chunks:
            return [], language or "und"
        self.load()
        rows = []
        detected = language
        for chunk in chunks:
            start, end = chunk["start"], chunk["end"]
            result = self._module.transcribe(
                waveform[start:end], path_or_hf_repo=self.model_path,
                language=detected, task="transcribe", temperature=0,
                condition_on_previous_text=False, verbose=None,
                initial_prompt=KOREAN_VOCABULARY if detected == "ko" else None,
            )
            detected = result["language"]
            for segment in result["segments"]:
                text = segment["text"].strip()
                if text:
                    rows.append({
                        "start": min(end / SAMPLE_RATE, start / SAMPLE_RATE + max(0, float(segment["start"]))),
                        "end": min(end / SAMPLE_RATE, start / SAMPLE_RATE + max(0, float(segment["end"]))),
                        "text": text,
                    })
        return rows, detected or "und"
