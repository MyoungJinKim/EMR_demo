#!/usr/bin/env python3
"""Download demo-only model copies without changing certification artifacts."""
from __future__ import annotations

import argparse
import platform
import shlex
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MLX_STT_MODEL = "mlx-community/whisper-large-v3-turbo"
MLX_STT_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
MLX_STT_REVISIONS = {
    MLX_STT_MODEL: MLX_STT_REVISION,
    "mlx-community/whisper-large-v3-mlx": "49e6aa286ad60c14352c404340ded53710378a11",
}
REVISIONS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "mlx-community/Qwen3.5-9B-4bit": "8b2b98c00a6b4d291155e4890773ca8f769aee53",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mlx-community/Qwen2.5-7B-Instruct-4bit": "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    default_model = (
        "mlx-community/Qwen3.5-9B-4bit"
        if apple_silicon
        else "Qwen/Qwen3.5-9B"
    )
    parser.add_argument("--stt-model", default=MLX_STT_MODEL if apple_silicon else "large-v3", choices=[*MLX_STT_REVISIONS, "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"])
    parser.add_argument("--emr-model", default=default_model, choices=[*REVISIONS, "Qwen/Qwen2.5-14B-Instruct"], help=f"default: {default_model}")
    args = parser.parse_args(argv)
    mlx_stt = args.stt_model in MLX_STT_REVISIONS
    if mlx_stt and not apple_silicon:
        parser.error("MLX STT는 Apple Silicon Mac에서만 사용할 수 있습니다. --stt-model large-v3를 사용하세요.")

    from huggingface_hub import snapshot_download

    cache = PROJECT_DIR / ".demo-cache"
    emr_dir = cache / args.emr_model.split("/")[-1]
    print(f"[1/2] STT 다운로드: {args.stt_model}", flush=True)
    if mlx_stt:
        stt_dir = snapshot_download(
            args.stt_model,
            revision=MLX_STT_REVISIONS[args.stt_model],
            local_dir=str(cache / args.stt_model.split("/")[-1]),
            allow_patterns=["config.json", "weights.npz", "weights.safetensors", "*.md", "LICENSE*"],
        )
    else:
        from faster_whisper import download_model

        stt_dir = download_model(args.stt_model, cache_dir=str(cache / "whisper"))
    print(f"[2/2] EMR 다운로드: {emr_dir}", flush=True)
    snapshot_download(
        args.emr_model,
        revision=REVISIONS.get(args.emr_model),
        local_dir=str(emr_dir),
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "*.md", "LICENSE*"],
    )
    print("\n준비 완료. 다음 설정으로 실행하세요:")
    print(f"export STT_MODEL={shlex.quote(str(stt_dir))}")
    print(f"export STT_DEVICE={'metal' if mlx_stt else 'cpu'}")
    print(f"export EMR_MODEL_PATH={shlex.quote(str(emr_dir))}")
    print("bash scripts/run_voice_demo.sh")


if __name__ == "__main__":
    main()
