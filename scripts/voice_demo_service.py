#!/usr/bin/env python3
"""Prepare, install, inspect, or remove this Mac's local demo LaunchAgent."""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import secrets
import socket
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LABEL = "org.nlpai.voice-demo"
URL = "http://127.0.0.1:8765/"


def configuration(project: Path, tunnel: bool = False) -> dict:
    config = {
        "Label": LABEL + (".tunnel" if tunnel else ""),
        "ProgramArguments": (
            [str(project / ".venv-demo/bin/python"), str(project / "scripts/run_voice_demo_tunnel.py")]
            if tunnel else ["/bin/bash", str(project / "scripts/run_voice_demo.sh")]
        ),
        "WorkingDirectory": str(project),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "Umask": 0o077,
        "StandardOutPath": str(project / f"output/demo/service/{'tunnel' if tunnel else 'server'}.stdout.log"),
        "StandardErrorPath": str(project / f"output/demo/service/{'tunnel' if tunnel else 'server'}.stderr.log"),
    }
    if not tunnel:
        config["EnvironmentVariables"] = {
            "DEMO_ACCESS_FILE": str(project / "output/demo/service/access.json"),
            "DEMO_PUBLIC_URL_FILE": str(project / "output/demo/service/public-url.txt"),
        }
    return config


def prepare_files(directory: Path, configs: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    access = directory / "access.json"
    if not access.exists():
        with access.open("x") as handle:
            access.chmod(0o600)
            json.dump({"username": "demo", "password": secrets.token_urlsafe(24)}, handle)
    credentials = json.loads(access.read_text())
    if (
        not isinstance(credentials, dict)
        or credentials.get("username") != "demo"
        or not isinstance(credentials.get("password"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{24,}", credentials["password"])
    ):
        raise RuntimeError("기존 access.json 설정이 올바르지 않습니다. 파일을 확인해 주세요.")
    access.chmod(0o600)
    for config in configs:
        (directory / f"{config['Label']}.plist").write_bytes(plistlib.dumps(config))


def check_owned(path: Path, config: dict) -> None:
    if path.exists():
        existing = plistlib.loads(path.read_bytes())
        if any(existing.get(key) != config[key] for key in ("Label", "ProgramArguments", "WorkingDirectory")):
            raise RuntimeError(f"다른 위치에서 설치한 서비스입니다. 덮어쓰거나 제거하지 않습니다: {path}")


def launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True, check=check, timeout=15)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "status", "uninstall"))
    action = parser.parse_args(argv).action
    configs = [configuration(PROJECT), configuration(PROJECT, tunnel=True)]
    directory = PROJECT / "output/demo/service"
    agent_directory = Path.home() / "Library/LaunchAgents"
    domain = f"gui/{os.getuid()}"

    if action == "prepare":
        prepare_files(directory, configs)
        print(f"검토용 설정: {directory}")
        print(f"접속 계정: demo / 비밀번호 파일: {directory / 'access.json'}")
        return
    if sys.platform != "darwin":
        raise RuntimeError("자동 실행 서비스는 macOS 사용자 로그인 세션에서만 사용할 수 있습니다.")

    states = []
    for config in configs:
        installed = agent_directory / f"{config['Label']}.plist"
        target = f"{domain}/{config['Label']}"
        check_owned(installed, config)
        running = launchctl("print", target, check=False)
        states.append((config, installed, target, running))
    if action == "status":
        print(f"로컬 주소: {URL}")
        public = directory / "public-url.txt"
        print(f"외부 주소: {public.read_text().strip() if public.exists() else '터널 연결 대기 중'}")
        print(f"접속 계정: demo / 비밀번호 파일: {directory / 'access.json'}")
        for config, installed, _, running in states:
            print(f"{config['Label']}: {'등록됨' if running.returncode == 0 else '등록되지 않음'} / {'자동 실행 설치됨' if installed.exists() else '설치되지 않음'}")
            for line in running.stdout.splitlines():
                if line.strip().startswith(("state =", "pid =", "last exit code =")):
                    print(line.strip())
        return
    for _, installed, _, running in states:
        if running.returncode == 0 and not installed.exists():
            raise RuntimeError("같은 이름의 서비스가 있지만 소유 설정을 확인할 수 없습니다. 변경하지 않습니다.")
    if action == "uninstall":
        for _, installed, target, running in reversed(states):
            if running.returncode == 0:
                launchctl("bootout", target)
            installed.unlink(missing_ok=True)
        print("자동 실행을 해제했습니다. 코드, 모델, 검증 결과와 로그는 유지합니다.")
        return
    if not os.access(PROJECT / ".venv-demo/bin/python", os.X_OK):
        raise RuntimeError("먼저 .venv-demo와 데모 모델을 준비해 주세요. DEMO.md의 설치 절차를 참고하세요.")
    if states[0][3].returncode != 0:
        with socket.socket() as listener:
            if listener.connect_ex(("127.0.0.1", 8765)) == 0:
                raise RuntimeError("8765 포트를 이미 사용 중입니다. 실행 중인 데모를 종료한 뒤 설치해 주세요. 기존 프로세스는 종료하지 않았습니다.")
    prepare_files(directory, configs)
    agent_directory.mkdir(parents=True, exist_ok=True)
    for config, installed, target, running in states:
        if running.returncode == 0:
            continue
        installed.write_bytes((directory / f"{config['Label']}.plist").read_bytes())
        installed.chmod(0o600)
        launchctl("enable", target)
        launchctl("bootstrap", domain, str(installed))
    print("서버와 외부 접속 터널의 자동 실행을 등록했습니다. status 명령으로 현재 주소를 확인하세요.")
    print("Mac 로그인·인터넷 연결·깨어 있는 상태가 필요합니다.")
    print("고정 URL 설정이 없으면 Quick Tunnel을 사용하며 재시작 시 주소가 변경됩니다.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"실패: {error}", file=sys.stderr)
        raise SystemExit(1)
