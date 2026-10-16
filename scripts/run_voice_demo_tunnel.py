#!/usr/bin/env python3
"""Run the demo's HTTPS tunnel, retaining its address when fixed mode is configured."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_URL = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com(?=[\s/|]|$)")
FIXED_URL = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*\.workers\.dev")


def write_url(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(url + "\n")
    temporary.replace(destination)


def record_url(line: str, destination: Path) -> str | None:
    match = PUBLIC_URL.search(line)
    if match:
        write_url(match[0], destination)
        return match[0]
    return None


def tunnel_command(binary: Path, directory: Path) -> tuple[list[str], str | None]:
    fixed = directory / "fixed-tunnel.json"
    base = [str(binary), "tunnel", "--no-autoupdate", "--loglevel", "info"]
    if not fixed.exists():
        return base + ["--url", "http://127.0.0.1:8765", "--http-host-header", "localhost:8765"], None
    try:
        config = json.loads(fixed.read_text())
        url = config["public_url"]
        if not isinstance(url, str) or not FIXED_URL.fullmatch(url):
            raise ValueError
        token = directory / "tunnel-token"
        if not token.is_file() or token.stat().st_size == 0 or token.stat().st_mode & 0o077:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError("고정 주소 설정 또는 비공개 tunnel-token 파일(권한 600)을 확인해 주세요.") from None
    # VPC Services require QUIC; the token stays out of process arguments and logs.
    return base + ["--protocol", "quic", "run", "--token-file", str(token)], url


def main() -> int:
    directory = PROJECT / "output/demo/service"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "public-url.txt"
    binary = next((path for path in (
        PROJECT / ".demo-tools/cloudflared", Path("/opt/homebrew/bin/cloudflared"), Path("/usr/local/bin/cloudflared")
    ) if os.access(path, os.X_OK)), None)
    if binary is None:
        raise RuntimeError("cloudflared 실행 파일이 없습니다.")
    command, fixed_url = tunnel_command(binary, directory)
    if fixed_url:
        write_url(fixed_url, destination)
    else:
        destination.unlink(missing_ok=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if not fixed_url:
                record_url(line, destination)
            sys.stderr.write(line)
        return process.wait()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if not fixed_url:
            destination.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"터널 실행 실패: {error}", file=sys.stderr)
        raise SystemExit(1)
