"""Local voice-to-EMR demo. Keeps the certification runner unchanged."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPBasic
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .demo_engine import DemoEngine

STATIC = Path(__file__).with_name("demo_static")
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 20_000
LIMITS = {"audio_mb": 25, "audio_seconds": 600, "text_chars": MAX_TEXT_CHARS}
AUDIO_TYPES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a", "audio/aac",
    "audio/flac", "audio/x-flac", "video/webm", "video/mp4", "application/octet-stream",
}


class NoteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    language: Literal["both", "ko", "en"] = "both"


def _valid_origin(origin: str, *, https_only: bool = False) -> bool:
    try:
        parsed = urlsplit(origin)
        return bool(
            origin and not any(character.isspace() or ord(character) < 32 for character in origin)
            and parsed.scheme in ({"https"} if https_only else {"http", "https"})
            and parsed.hostname and parsed.port != 0 and not parsed.netloc.endswith(":")
            and not (parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment)
            and "@" not in parsed.netloc and "\\" not in origin
            and origin == f"{parsed.scheme}://{parsed.netloc}"
        )
    except ValueError:
        return False


def _security_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    return response


def create_app(engine: DemoEngine | None = None) -> FastAPI:
    access_file = os.environ.get("DEMO_ACCESS_FILE")
    public_url_file = os.environ.get("DEMO_PUBLIC_URL_FILE")
    access = None
    if access_file is not None:
        try:
            configuration = json.loads(Path(access_file).read_text())
            if (
                not isinstance(configuration, dict) or configuration.get("username") != "demo"
                or not isinstance(configuration.get("password"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{24,}", configuration["password"])
            ):
                raise ValueError
            access = (b"demo", configuration["password"].encode("ascii"))
        except (OSError, UnicodeError, ValueError):
            raise RuntimeError("데모 접근 인증 설정을 확인하세요.") from None
    authentication = HTTPBasic(realm="NLP-AI Demo", auto_error=False)
    app = FastAPI(title="NLP&AI · Voice to SOAP", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    service = engine if engine is not None else DemoEngine()
    inference_lock = asyncio.Lock()

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if access is not None:
            try:
                credentials = await authentication(request)
            except HTTPException:
                credentials = None
            username_valid = secrets.compare_digest(credentials.username.encode("utf8") if credentials else b"", access[0])
            password_valid = secrets.compare_digest(credentials.password.encode("utf8") if credentials else b"", access[1])
            if not (username_valid and password_valid):
                return _security_headers(JSONResponse(
                    {"detail": "데모 접근 인증이 필요합니다."}, status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="NLP-AI Demo"'},
                ))
        if request.method == "POST":
            origin = request.headers.get("origin")
            origin_allowed = bool(origin and _valid_origin(origin) and urlsplit(origin).netloc == request.headers.get("host"))
            if origin and not origin_allowed and access is not None and public_url_file:
                try:
                    # A quick-tunnel restart changes the origin without restarting the model server.
                    public_origin = Path(public_url_file).read_text().strip()
                    origin_allowed = _valid_origin(public_origin, https_only=True) and origin == public_origin
                except (OSError, UnicodeError):
                    origin_allowed = False
            if not origin_allowed and (origin is not None or access is not None):
                return _security_headers(JSONResponse({"detail": "이 데모 화면에서 다시 요청해 주세요."}, status_code=403))
            content_length = request.headers.get("content-length")
            limit = MAX_AUDIO_BYTES if request.url.path == "/api/transcribe" else 128 * 1024
            if content_length:
                try:
                    invalid = int(content_length) < 0 or int(content_length) > limit
                except ValueError:
                    invalid = True
                if invalid:
                    return _security_headers(JSONResponse({"detail": "요청 크기가 허용 범위를 초과했습니다."}, status_code=413))
        response = await call_next(request)
        return _security_headers(response)

    async def infer(operation, *args):
        if inference_lock.locked():
            raise HTTPException(409, "다른 작업이 진행 중입니다. 완료 후 다시 시도해 주세요.")
        async with inference_lock:
            try:
                return await run_in_threadpool(operation, *args)
            except HTTPException:
                raise
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
            except (RuntimeError, FileNotFoundError, ImportError) as error:
                raise HTTPException(503, str(error)) from error
            except Exception as error:
                # Never put transcripts, audio, credentials, or provider responses in logs.
                logging.getLogger(__name__).error("Inference failed (%s)", type(error).__name__)
                raise HTTPException(500, "처리하지 못했습니다. 모델 설정을 확인한 뒤 다시 시도해 주세요.") from error

    def health():
        return {"status": "ok", **service.health(), "limits": LIMITS, "busy": inference_lock.locked()}

    @app.get("/api/health")
    def get_health():
        return health()

    @app.post("/api/warmup")
    async def warmup():
        await infer(service.warmup)
        return health()

    @app.post("/api/transcribe")
    async def transcribe(request: Request, language: Literal["ko", "en", "auto"] = "ko"):
        if request.headers.get("content-type", "").split(";")[0].strip().lower() not in AUDIO_TYPES:
            raise HTTPException(415, "WAV, MP3, M4A, WebM 등 오디오 파일을 선택해 주세요.")
        size = 0
        with tempfile.TemporaryDirectory(prefix="scribe-audio-") as directory:
            path = Path(directory) / "recording.audio"
            with path.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_AUDIO_BYTES:
                        raise HTTPException(413, "오디오 파일은 최대 25 MB까지 사용할 수 있습니다.")
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, "녹음된 오디오가 없습니다. 다시 녹음해 주세요.")
            return await infer(service.transcribe, path, language)

    @app.post("/api/generate")
    async def generate(request: Request):
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "JSON 형식의 전사문이 필요합니다.")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 128 * 1024:
                raise HTTPException(413, "전사문이 너무 깁니다.")
        try:
            note_request = NoteRequest.model_validate_json(body)
        except ValueError:
            raise HTTPException(422, "전사문은 1~20,000자, 언어는 both/ko/en이어야 합니다.")
        text = note_request.text.strip()
        if not text:
            raise HTTPException(400, "전사문을 입력하거나 먼저 음성을 녹음해 주세요.")

        def make_notes():
            started = time.perf_counter()
            languages = ("ko", "en") if note_request.language == "both" else (note_request.language,)
            english = service.generate(text, "en")
            generated = {"en": english}
            if "ko" in languages:
                generated["ko"] = service.translate_to_korean(english["note"], text)
            results = {language: generated[language] for language in languages}
            first = results[languages[0]]
            return {
                "notes": {language: result["note"] for language, result in results.items()},
                "note": first["note"],
                "model": first["model"],
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "finish_reason": "length" if any(result.get("finish_reason") == "length" for result in results.values()) else "stop",
                "warnings": [f"{'한국어' if language == 'ko' else 'English'}: {warning}" for language, result in results.items() for warning in result.get("warnings", [])],
            }

        return await infer(make_notes)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/static/{filename}")
    def static_file(filename: str):
        if filename not in {"app.css", "app.js", "nlp-ai-logo.png"}:
            raise HTTPException(404)
        return FileResponse(STATIC / filename)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Local microphone → Whisper → bilingual Qwen EMR demo")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
