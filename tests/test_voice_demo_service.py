from __future__ import annotations

import importlib.util
import json
import plistlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/voice_demo_service.py"
SPEC = importlib.util.spec_from_file_location("voice_demo_service", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


def test_prepare_only_writes_reviewable_workspace_plist(tmp_path, monkeypatch):
    project = tmp_path / "NLP&AI Lab 한국어"
    monkeypatch.setattr(service, "PROJECT", project)
    monkeypatch.setattr(service, "launchctl", lambda *a, **k: pytest.fail("prepare must not activate a service"))
    service.main(["prepare"])
    path = project / "output/demo/service" / f"{service.LABEL}.plist"
    config = plistlib.loads(path.read_bytes())
    assert config["ProgramArguments"] == ["/bin/bash", str(project / "scripts/run_voice_demo.sh")]
    assert config["WorkingDirectory"] == str(project)
    assert config["KeepAlive"] and config["RunAtLoad"]
    assert config["Umask"] == 0o077
    assert config["EnvironmentVariables"]["DEMO_ACCESS_FILE"] == str(project / "output/demo/service/access.json")
    assert "password" not in path.read_text()
    tunnel = path.with_name(service.LABEL + ".tunnel.plist")
    assert plistlib.loads(tunnel.read_bytes())["ProgramArguments"][1] == str(project / "scripts/run_voice_demo_tunnel.py")


def test_prepare_keeps_existing_password_private_and_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(service, "PROJECT", tmp_path)
    service.main(["prepare"])
    access = tmp_path / "output/demo/service/access.json"
    original = access.read_bytes()
    password = json.loads(original)["password"]
    assert len(password) >= 24
    assert access.stat().st_mode & 0o777 == 0o600
    service.main(["prepare"])
    assert access.read_bytes() == original
    assert password not in capsys.readouterr().out


@pytest.mark.parametrize("password", ["a" * 23, "a" * 24 + ":", "a" * 24 + "\n", "가" * 24, None])
def test_prepare_rejects_passwords_the_server_would_reject(tmp_path, password):
    directory = tmp_path / "service"
    directory.mkdir()
    access = directory / "access.json"
    original = json.dumps({"username": "demo", "password": password}).encode()
    access.write_bytes(original)
    with pytest.raises(RuntimeError, match="올바르지"):
        service.prepare_files(directory, [service.configuration(tmp_path)])
    assert access.read_bytes() == original
    assert not list(directory.glob("*.plist"))


def test_does_not_overwrite_or_remove_another_workspace_service(tmp_path):
    path = tmp_path / "service.plist"
    path.write_bytes(plistlib.dumps(service.configuration(tmp_path / "other")))
    with pytest.raises(RuntimeError, match="다른 위치"):
        service.check_owned(path, service.configuration(tmp_path / "current"))


def test_matching_existing_service_is_owned(tmp_path):
    config = service.configuration(tmp_path)
    path = tmp_path / "service.plist"
    path.write_bytes(plistlib.dumps(config))
    service.check_owned(path, config)


def test_install_rejects_occupied_port_without_writing_or_killing(tmp_path, monkeypatch):
    project = tmp_path / "project"
    home = tmp_path / "home"
    monkeypatch.setattr(service, "PROJECT", project)
    monkeypatch.setattr(service.Path, "home", lambda: home)
    monkeypatch.setattr(service.sys, "platform", "darwin")
    monkeypatch.setattr(service.os, "access", lambda *args: True)
    calls = []

    def ctl(*args, **kwargs):
        calls.append(args)
        return service.subprocess.CompletedProcess(args, 1, "", "not loaded")

    class OccupiedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def connect_ex(self, address):
            assert address == ("127.0.0.1", 8765)
            return 0

    monkeypatch.setattr(service, "launchctl", ctl)
    monkeypatch.setattr(service.socket, "socket", OccupiedSocket)
    with pytest.raises(RuntimeError, match="이미 사용 중"):
        service.main(["install"])
    assert len(calls) == 2 and all(call[0] == "print" for call in calls)
    assert not project.exists() and not home.exists()


def test_public_url_requires_real_quick_tunnel_hostname(tmp_path):
    tunnel_spec = importlib.util.spec_from_file_location("run_voice_demo_tunnel", SCRIPT.with_name("run_voice_demo_tunnel.py"))
    assert tunnel_spec is not None and tunnel_spec.loader is not None
    tunnel = importlib.util.module_from_spec(tunnel_spec)
    tunnel_spec.loader.exec_module(tunnel)
    destination = tmp_path / "public-url.txt"
    assert tunnel.record_url("https://demo.trycloudflare.com.evil.example", destination) is None
    assert not destination.exists()
    assert tunnel.record_url("INF | https://test-voice-123.trycloudflare.com |", destination) == "https://test-voice-123.trycloudflare.com"
    assert destination.read_text().strip() == "https://test-voice-123.trycloudflare.com"
