import base64
import json
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from mediqa_qwen_eval.demo import create_app


@pytest.fixture(autouse=True)
def local_demo_environment(monkeypatch):
    monkeypatch.delenv("DEMO_ACCESS_FILE", raising=False)
    monkeypatch.delenv("DEMO_PUBLIC_URL_FILE", raising=False)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.audio_path = None

    def health(self):
        return {"stt": {"model": "whisper-test", "loaded": True}, "emr": {"model": "qwen-test", "loaded": True, "backend": "local"}}

    def warmup(self):
        return self.health()

    def transcribe(self, path, language):
        self.audio_path = Path(path)
        assert self.audio_path.read_bytes() == b"fake-audio"
        return {"text": "사흘 동안 기침", "segments": [], "language": language, "duration_seconds": 3, "elapsed_seconds": 0.1, "model": "whisper-test"}

    def generate(self, text, language):
        self.calls.append((text, language))
        return {"note": f"{language}: {text}", "model": "qwen-test", "finish_reason": "stop", "warnings": []}

    def translate_to_korean(self, english_note, source_text):
        self.calls.append((english_note, "translate_to_korean", source_text))
        return {"note": f"ko: {english_note}", "model": "qwen-test", "finish_reason": "stop", "warnings": []}


def test_supplied_logo_is_served_without_exposing_other_files():
    with TestClient(create_app(FakeEngine())) as client:
        logo = client.get("/static/nlp-ai-logo.png")
        assert logo.status_code == 200
        assert logo.headers["content-type"] == "image/png"
        assert logo.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert client.get("/static/secret.txt").status_code == 404


def test_both_notes_translate_generated_english_with_original_vocabulary_reference():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/api/generate", json={"text": " 사흘 동안 기침 "})
    assert response.status_code == 200
    data = response.json()
    assert data["notes"] == {"ko": "ko: en: 사흘 동안 기침", "en": "en: 사흘 동안 기침"}
    assert list(data["notes"]) == ["ko", "en"]
    assert data["note"] == data["notes"]["ko"]
    assert engine.calls == [("사흘 동안 기침", "en"), ("en: 사흘 동안 기침", "translate_to_korean", "사흘 동안 기침")]


@pytest.mark.parametrize("language", ["ko", "en"])
def test_single_language_response_keeps_english_as_the_generation_source(language):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/api/generate", json={"text": "기침", "language": language})
    assert response.status_code == 200
    data = response.json()
    assert list(data["notes"]) == [language]
    assert data["note"] == data["notes"][language]
    assert engine.calls == ([("기침", "en")] if language == "en" else [
        ("기침", "en"), ("en: 기침", "translate_to_korean", "기침"),
    ])


@pytest.mark.parametrize("stage", ["generate", "translate_to_korean"])
def test_generation_or_translation_failure_does_not_return_partial_notes(stage):
    engine = FakeEngine()

    def fail(*args):
        raise RuntimeError("SOAP 생성 실패")

    setattr(engine, stage, fail)
    with TestClient(create_app(engine)) as client:
        response = client.post("/api/generate", json={"text": "기침"})
    assert response.status_code == 503
    assert response.json() == {"detail": "SOAP 생성 실패"}
    assert engine.calls == ([("기침", "en")] if stage == "translate_to_korean" else [])


def test_upload_is_deleted_after_transcription():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/api/transcribe?language=ko", content=b"fake-audio", headers={"Content-Type": "audio/webm;codecs=opus"})
    assert response.status_code == 200
    assert response.json()["text"] == "사흘 동안 기침"
    assert not engine.audio_path.exists()


def test_bad_inputs_and_cross_origin_are_rejected():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.post("/api/generate", json={"text": " "}).status_code == 400
        assert client.post("/api/generate", json={"text": "a" * 20001}).status_code == 422
        assert client.post("/api/generate", json={"text": "기침", "language": "fr"}).status_code == 422
        assert client.post("/api/generate", content="{}", headers={"Content-Type": "text/plain"}).status_code == 415
        assert client.post("/api/generate", json={"text": "기침"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/warmup", headers={"Origin": "null"}).status_code == 403
        assert client.post("/api/transcribe", content=b"", headers={"Content-Type": "audio/wav"}).status_code == 400
        assert client.post("/api/transcribe", content=b"abc", headers={"Content-Type": "text/html"}).status_code == 415
        assert client.post("/api/transcribe", content=b"abc", headers={"Content-Type": "audio/wav", "Content-Length": str(26 * 1024 * 1024)}).status_code == 413
        assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
    assert not engine.calls


def test_failure_cleans_upload_and_does_not_leak_unexpected_error():
    engine = FakeEngine()

    def fail(path, language):
        engine.audio_path = Path(path)
        raise Exception("private transcript and secret provider response")

    engine.transcribe = fail
    with TestClient(create_app(engine)) as client:
        response = client.post("/api/transcribe", content=b"fake-audio", headers={"Content-Type": "audio/wav"})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert not engine.audio_path.exists()


def test_length_warning_preserves_language():
    engine = FakeEngine()
    engine.generate = lambda text, language: {"note": language, "model": "test", "finish_reason": "length", "warnings": ["출력이 잘렸습니다."]}
    engine.translate_to_korean = lambda english, source: {"note": "ko", "model": "test", "finish_reason": "length", "warnings": ["출력이 잘렸습니다."]}
    with TestClient(create_app(engine)) as client:
        data = client.post("/api/generate", json={"text": "기침"}).json()
    assert data["finish_reason"] == "length"
    assert data["warnings"] == ["한국어: 출력이 잘렸습니다.", "English: 출력이 잘렸습니다."]


def test_parallel_generation_is_rejected_but_health_responds():
    engine = FakeEngine()
    entered, release = Event(), Event()

    def slow(text, language):
        entered.set()
        assert release.wait(5)
        return {"note": language, "model": "test", "warnings": []}

    engine.generate = slow
    with TestClient(create_app(engine)) as client:
        worker = Thread(target=lambda: client.post("/api/generate", json={"text": "기침"}))
        worker.start()
        try:
            assert entered.wait(5)
            assert client.get("/api/health").json()["busy"] is True
            assert client.post("/api/generate", json={"text": "다른 환자"}).status_code == 409
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()


@pytest.fixture
def protected_demo(monkeypatch, tmp_path):
    password = "test-only-password_0123456789"
    access_file = tmp_path / "access.json"
    access_file.write_text(json.dumps({"username": "demo", "password": password}))
    public_url_file = tmp_path / "public-url.txt"
    public_url_file.write_text("https://current-demo.example\n")
    monkeypatch.setenv("DEMO_ACCESS_FILE", str(access_file))
    monkeypatch.setenv("DEMO_PUBLIC_URL_FILE", str(public_url_file))
    return {"password": password, "access_file": access_file, "public_url_file": public_url_file}


def assert_security_headers(response):
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_public_auth_covers_static_health_and_post_without_exposing_credentials(protected_demo):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        for method, route in [("GET", "/"), ("GET", "/api/health"), ("GET", "/static/app.js"),
                              ("GET", "/openapi.json"), ("POST", "/api/generate"), ("POST", "/api/transcribe")]:
            response = client.request(method, route)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == 'Basic realm="NLP-AI Demo"'
            assert protected_demo["password"] not in response.text
            assert str(protected_demo["access_file"]) not in response.text
            assert_security_headers(response)
        for credentials in [("demo", "wrong"), ("wrong", protected_demo["password"])]:
            response = client.get("/api/health", auth=credentials)
            assert response.status_code == 401
            assert_security_headers(response)
        client.auth = ("demo", protected_demo["password"])
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        health = client.get("/api/health")
        assert health.status_code == 200
        assert protected_demo["password"] not in health.text
        assert str(protected_demo["access_file"]) not in health.text
        assert client.get("/api/health", headers={"Host": "external.example"}).status_code == 400
    assert not engine.calls
    assert engine.audio_path is None


@pytest.mark.parametrize("authorization", [
    "", "Bearer token", "Basic invalid", "Basic !!!", "Basic " + base64.b64encode(b"no-colon").decode(),
    "Basic " + base64.b64encode("사용자:비밀".encode()).decode(),
])
def test_malformed_basic_auth_returns_protected_401(protected_demo, authorization):
    with TestClient(create_app(FakeEngine())) as client:
        response = client.get("/api/health", headers={"Authorization": authorization})
    assert response.status_code == 401
    assert_security_headers(response)


def test_public_post_requires_exact_current_origin_and_keeps_local_origin(protected_demo):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        client.auth = ("demo", protected_demo["password"])
        for origin in [None, "", "null", "https://evil.example", "https://current-demo.example.evil.example",
                       "http://current-demo.example", "https://current-demo.example/path", "https://current-demo.example/"]:
            headers = {} if origin is None else {"Origin": origin}
            response = client.post("/api/generate", json={"text": "기침"}, headers=headers)
            assert response.status_code == 403
            assert_security_headers(response)
        assert not engine.calls
        for origin in ["http://testserver", "https://current-demo.example"]:
            assert client.post("/api/generate", json={"text": "기침"}, headers={"Origin": origin}).status_code == 200
        protected_demo["public_url_file"].write_text("https://restarted-demo.example\n")
        assert client.post("/api/warmup", headers={"Origin": "https://current-demo.example"}).status_code == 403
        assert client.post("/api/warmup", headers={"Origin": "https://restarted-demo.example"}).status_code == 200
        response = client.post("/api/transcribe", content=b"fake-audio", headers={
            "Origin": "https://restarted-demo.example", "Content-Type": "audio/wav", "Content-Length": str(26 * 1024 * 1024),
        })
        assert response.status_code == 413
        assert_security_headers(response)


@pytest.mark.parametrize("public_url", [None, "", "http://current-demo.example", "https://current-demo.example/",
    "https://demo:secret@current-demo.example", "https://current-demo.example?", "https://current-demo.example#",
    "https://current-demo.example:bad", "https://[bad", "https://current-demo.example:\n", "https://current-demo.example\nevil"])
def test_bad_or_missing_public_url_fails_closed(protected_demo, public_url):
    path = protected_demo["public_url_file"]
    if public_url is None:
        path.unlink()
    else:
        path.write_text(public_url)
    with TestClient(create_app(FakeEngine())) as client:
        client.auth = ("demo", protected_demo["password"])
        response = client.post("/api/warmup", headers={"Origin": public_url.strip() if public_url and "\n" not in public_url else "https://current-demo.example"})
        assert response.status_code == 403
        assert_security_headers(response)
        assert client.post("/api/warmup", headers={"Origin": "http://testserver"}).status_code == 200


def test_public_origin_is_not_enabled_without_auth(monkeypatch, tmp_path):
    path = tmp_path / "public.txt"
    path.write_text("https://current-demo.example")
    monkeypatch.setenv("DEMO_PUBLIC_URL_FILE", str(path))
    with TestClient(create_app(FakeEngine())) as client:
        assert client.post("/api/warmup", headers={"Origin": "https://current-demo.example"}).status_code == 403
        assert client.post("/api/warmup").status_code == 200


@pytest.mark.parametrize("configuration", [None, "not-json", "[]", "{}", '{"username":"demo","password":"short"}',
    '{"username":"demo","password":123}', '{"username":"demo","password":"spaces are not a URLsafe secret"}'])
def test_invalid_access_config_prevents_startup_without_leaking_details(monkeypatch, tmp_path, configuration):
    path = tmp_path / "private-access.json"
    if configuration is not None:
        path.write_text(configuration)
    monkeypatch.setenv("DEMO_ACCESS_FILE", str(path))
    with pytest.raises(RuntimeError, match="접근 인증 설정") as error:
        create_app(FakeEngine())
    assert str(path) not in str(error.value)
    assert "password" not in str(error.value)


def test_access_credentials_are_loaded_once(protected_demo):
    application = create_app(FakeEngine())
    protected_demo["access_file"].write_text("invalid after startup")
    with TestClient(application) as client:
        assert client.get("/api/health", auth=("demo", protected_demo["password"])).status_code == 200
