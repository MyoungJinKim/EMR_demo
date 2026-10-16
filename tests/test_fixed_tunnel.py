from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_voice_demo_tunnel.py"
SPEC = importlib.util.spec_from_file_location("fixed_tunnel", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
URL = "https://nlpai-voice-demo.example-account.workers.dev"


@pytest.fixture
def configured(tmp_path):
    directory = tmp_path / "output/demo/service"
    directory.mkdir(parents=True)
    (directory / "fixed-tunnel.json").write_text(json.dumps({"public_url": URL}))
    token = directory / "tunnel-token"
    token.write_text("private-tunnel-token")
    token.chmod(0o600)
    return directory


def test_named_tunnel_uses_private_token_file_not_secret_in_arguments(configured):
    command, url = module.tunnel_command(Path("cloudflared"), configured)
    assert url == URL
    assert command[-4:] == ["quic", "run", "--token-file", str(configured / "tunnel-token")]
    assert "--protocol" in command
    assert "private-tunnel-token" not in " ".join(command)
    assert "--url" not in command


@pytest.mark.parametrize("value", [
    "https://demo.example.workers.dev.evil.com", "http://demo.example.workers.dev",
    "https://demo.example.workers.dev/", "https://demo.example.workers.dev\n",
    "https://user:password@demo.example.workers.dev", None,
])
def test_invalid_fixed_origin_fails_without_quick_tunnel_fallback(configured, value):
    (configured / "fixed-tunnel.json").write_text(json.dumps({"public_url": value}))
    with pytest.raises(RuntimeError, match="고정 주소 설정"):
        module.tunnel_command(Path("cloudflared"), configured)


@pytest.mark.parametrize("contents", ["invalid secret JSON", "{}", "[]", "null"])
def test_bad_config_does_not_echo_its_contents(configured, contents):
    (configured / "fixed-tunnel.json").write_text(contents)
    with pytest.raises(RuntimeError) as caught:
        module.tunnel_command(Path("cloudflared"), configured)
    assert contents not in str(caught.value)


@pytest.mark.parametrize("problem", ["missing", "empty", "public"])
def test_bad_token_file_fails_closed(configured, problem):
    token = configured / "tunnel-token"
    if problem == "missing":
        token.unlink()
    elif problem == "empty":
        token.write_text("")
    else:
        token.chmod(0o644)
    with pytest.raises(RuntimeError, match="권한 600"):
        module.tunnel_command(Path("cloudflared"), configured)


def test_fixed_url_survives_exit_and_restart_and_ignores_quick_url_logs(configured, monkeypatch):
    monkeypatch.setattr(module, "PROJECT", configured.parents[2])
    monkeypatch.setattr(module.os, "access", lambda *args: True)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = iter(["INF https://wrong-random.trycloudflare.com\n"])

        def wait(self, **kwargs):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    for _ in range(2):
        assert module.main() == 0
        assert (configured / "public-url.txt").read_text().strip() == URL
    assert commands[0] == commands[1]


def test_unconfigured_tunnel_keeps_quick_tunnel_behavior(tmp_path):
    command, url = module.tunnel_command(Path("cloudflared"), tmp_path)
    assert url is None
    assert command[-4:] == ["--url", "http://127.0.0.1:8765", "--http-host-header", "localhost:8765"]
