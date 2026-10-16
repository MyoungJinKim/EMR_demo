from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_voice_demo.py"
SPEC = importlib.util.spec_from_file_location("prepare_voice_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.mark.parametrize("system,machine,args,stt,model,revision", [
    ("Darwin", "arm64", [], prepare.MLX_STT_MODEL, "mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"),
    ("Linux", "x86_64", [], "large-v3", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Darwin", "x86_64", [], "large-v3", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Darwin", "arm64", ["--stt-model", "small", "--emr-model", "Qwen/Qwen2.5-1.5B-Instruct"], "small", "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    ("Darwin", "arm64", ["--stt-model", "large-v3", "--emr-model", "Qwen/Qwen2.5-7B-Instruct"], "large-v3", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    ("Darwin", "arm64", ["--emr-model", "mlx-community/Qwen2.5-7B-Instruct-4bit"], prepare.MLX_STT_MODEL, "mlx-community/Qwen2.5-7B-Instruct-4bit", "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"),
    ("Darwin", "arm64", ["--stt-model", "mlx-community/whisper-large-v3-mlx"], "mlx-community/whisper-large-v3-mlx", "mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"),
])
def test_preparation_keeps_models_in_demo_cache(tmp_path, capsys, monkeypatch, system, machine, args, stt, model, revision):
    monkeypatch.setattr(prepare.platform, "system", lambda: system)
    monkeypatch.setattr(prepare.platform, "machine", lambda: machine)
    whisper_download = Mock(return_value=str(tmp_path / ".demo-cache/whisper/model-snapshot"))
    qwen_download = Mock(side_effect=lambda *args, **kwargs: kwargs["local_dir"])
    with patch.object(prepare, "PROJECT_DIR", tmp_path), patch.dict(
        sys.modules,
        {
            "faster_whisper": SimpleNamespace(download_model=whisper_download),
            "huggingface_hub": SimpleNamespace(snapshot_download=qwen_download),
        },
    ):
        prepare.main(args)

    if stt in prepare.MLX_STT_REVISIONS:
        whisper_download.assert_not_called()
        assert qwen_download.call_count == 2
        stt_call = qwen_download.call_args_list[0]
        assert stt_call.args == (stt,)
        assert stt_call.kwargs["revision"] == prepare.MLX_STT_REVISIONS[stt]
        assert stt_call.kwargs["local_dir"] == str(tmp_path / ".demo-cache" / stt.split("/")[-1])
        assert {"config.json", "weights.npz", "weights.safetensors"}.issubset(stt_call.kwargs["allow_patterns"])
    else:
        whisper_download.assert_called_once_with(stt, cache_dir=str(tmp_path / ".demo-cache/whisper"))
        assert qwen_download.call_count == 1
    assert qwen_download.call_args.args == (model,)
    assert qwen_download.call_args.kwargs["revision"] == revision
    assert qwen_download.call_args.kwargs["local_dir"] == str(tmp_path / ".demo-cache" / model.split("/")[-1])
    assert {"*.jinja", "*.json", "*.safetensors"}.issubset(qwen_download.call_args.kwargs["allow_patterns"])
    output = capsys.readouterr().out
    assert "export STT_MODEL=" in output
    assert f"export STT_DEVICE={'metal' if stt in prepare.MLX_STT_REVISIONS else 'cpu'}" in output


def test_preparation_and_runtime_share_pinned_turbo_default():
    from mediqa_qwen_eval import demo_stt

    assert prepare.MLX_STT_MODEL == demo_stt.MLX_STT_MODEL == "mlx-community/whisper-large-v3-turbo"
    assert prepare.MLX_STT_REVISION == demo_stt.MLX_STT_REVISION == "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"


def test_download_failure_does_not_claim_ready(tmp_path, capsys):
    qwen_download = Mock()
    with patch.object(prepare, "PROJECT_DIR", tmp_path), patch.dict(
        sys.modules,
        {
            "faster_whisper": SimpleNamespace(download_model=Mock(side_effect=OSError("network unavailable"))),
            "huggingface_hub": SimpleNamespace(snapshot_download=qwen_download),
        },
    ):
        with pytest.raises(OSError):
            prepare.main(["--stt-model", "large-v3", "--emr-model", "Qwen/Qwen2.5-7B-Instruct"])

    qwen_download.assert_not_called()
    assert "준비 완료" not in capsys.readouterr().out


def test_mlx_download_failure_does_not_download_emr(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(prepare.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(prepare.platform, "machine", lambda: "arm64")
    download = Mock(side_effect=OSError("network unavailable"))
    with patch.object(prepare, "PROJECT_DIR", tmp_path), patch.dict(
        sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)},
    ):
        with pytest.raises(OSError):
            prepare.main([])
    download.assert_called_once()
    assert download.call_args.args == (prepare.MLX_STT_MODEL,)
    assert "준비 완료" not in capsys.readouterr().out


@pytest.mark.parametrize("model", prepare.MLX_STT_REVISIONS)
def test_mlx_stt_on_unsupported_device_fails_before_download(monkeypatch, model):
    monkeypatch.setattr(prepare.platform, "system", lambda: "Linux")
    monkeypatch.setattr(prepare.platform, "machine", lambda: "x86_64")
    with pytest.raises(SystemExit) as error:
        prepare.main(["--stt-model", model])
    assert error.value.code == 2
