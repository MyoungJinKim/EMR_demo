import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mediqa_qwen_eval import demo_stt
from mediqa_qwen_eval.demo_stt import MlxWhisper, is_mlx_whisper_model


WHISPER_CONFIG = {
    "n_mels": 128, "n_audio_ctx": 1500, "n_audio_state": 1280, "n_audio_head": 20,
    "n_audio_layer": 32, "n_vocab": 51866, "n_text_ctx": 448, "n_text_state": 1280,
    "n_text_head": 20, "n_text_layer": 4, "model_type": "whisper",
}


@pytest.mark.parametrize("weight_name", ["weights.npz", "weights.safetensors"])
def test_local_mlx_formats_load_once_without_repository_lookup(tmp_path, monkeypatch, weight_name):
    (tmp_path / "config.json").write_text(json.dumps(WHISPER_CONFIG))
    (tmp_path / weight_name).write_bytes(b"stub weights")
    calls = []
    core = SimpleNamespace(float16="float16")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    module = SimpleNamespace(ModelHolder=SimpleNamespace(get_model=lambda *args: calls.append(args)))
    monkeypatch.setattr(demo_stt, "import_module", lambda name: module)

    assert is_mlx_whisper_model(tmp_path)
    model = MlxWhisper(tmp_path)
    model.load()
    model.load()
    assert calls == [(str(tmp_path.resolve()), "float16")]


@pytest.mark.parametrize("config", [
    "{broken", "[]", json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4}}),
    json.dumps({"model_type": "whisper"}), json.dumps({**WHISPER_CONFIG, "n_text_layer": 0}),
])
def test_invalid_or_other_model_config_rejected_offline(tmp_path, config):
    (tmp_path / "config.json").write_text(config)
    (tmp_path / "weights.safetensors").write_bytes(b"stub weights")
    assert not is_mlx_whisper_model(tmp_path)
    with pytest.raises(ValueError, match="모델 파일"):
        MlxWhisper(tmp_path).load()


def test_legacy_mlx_config_without_model_type_is_supported(tmp_path):
    config = {key: value for key, value in WHISPER_CONFIG.items() if key != "model_type"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert not is_mlx_whisper_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"unsupported filename")
    assert not is_mlx_whisper_model(tmp_path)
    (tmp_path / "weights.npz").write_bytes(b"stub weights")
    assert is_mlx_whisper_model(tmp_path)


def test_missing_model_stays_offline(tmp_path):
    with pytest.raises(ValueError, match="모델 파일"):
        MlxWhisper(tmp_path / "missing").load()


def test_silence_does_not_load_or_hallucinate(tmp_path, monkeypatch):
    from faster_whisper import vad

    monkeypatch.setattr(vad, "get_speech_timestamps", lambda *args, **kwargs: [])
    model = MlxWhisper(tmp_path)
    monkeypatch.setattr(model, "load", lambda: pytest.fail("silent audio loaded model"))
    assert model.transcribe(np.zeros(16000), None) == ([], "und")


def test_precomputed_speech_chunks_skip_duplicate_vad(tmp_path, monkeypatch):
    from mediqa_qwen_eval import demo_stt

    monkeypatch.setattr(demo_stt, "speech_chunks", lambda *_: pytest.fail("VAD ran twice"))
    model = MlxWhisper(tmp_path)
    received = []

    def decode(waveform, **kwargs):
        received.append(len(waveform))
        return {"language": "ko", "segments": [{"start": 0, "end": 1, "text": "기침"}]}

    model._module = SimpleNamespace(transcribe=decode)
    rows, language = model.transcribe(np.zeros(32000), "ko", chunks=[{"start": 16000, "end": 32000}])
    assert received == [16000]
    assert rows == [{"start": 1.0, "end": 2.0, "text": "기침"}]
    assert language == "ko"
    assert model.transcribe(np.zeros(32000), "ko", chunks=[]) == ([], "ko")


def test_vad_offsets_preserve_original_timeline(tmp_path, monkeypatch):
    from faster_whisper import vad

    chunks = [{"start": 16000, "end": 48000}, {"start": 80000, "end": 96000}]
    monkeypatch.setattr(vad, "get_speech_timestamps", lambda *args, **kwargs: chunks)
    calls = []

    def decode(waveform, **kwargs):
        calls.append(kwargs)
        return {"language": "ko", "segments": [
            {"start": 0.2, "end": 9, "text": " 복용약 없음. "},
            {"start": 1, "end": 1, "text": " "},
        ]}

    model = MlxWhisper(tmp_path)
    model._module = SimpleNamespace(transcribe=decode)
    rows, language = model.transcribe(np.zeros(100000), None)
    assert rows == [
        {"start": 1.2, "end": 3.0, "text": "복용약 없음."},
        {"start": 5.2, "end": 6.0, "text": "복용약 없음."},
    ]
    assert language == "ko"
    assert calls[0]["language"] is None
    assert calls[1]["language"] == "ko"
    assert calls[0]["initial_prompt"] is None
    assert "알레르기" in calls[1]["initial_prompt"]


def test_unload_releases_global_gpu_model(tmp_path, monkeypatch):
    import sys

    cache_cleared = []
    core = SimpleNamespace(clear_cache=lambda: cache_cleared.append(True))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    model = MlxWhisper(tmp_path)
    holder = SimpleNamespace(model=object(), model_path=model.model_path)
    model._module = SimpleNamespace(ModelHolder=holder)
    model.unload()
    assert holder.model is None
    assert holder.model_path is None
    assert model._module is None
    assert cache_cleared == [True]
