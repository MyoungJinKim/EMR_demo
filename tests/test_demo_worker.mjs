import assert from "node:assert/strict";
import test from "node:test";
import worker from "../deploy/cloudflare/worker.mjs";

const origin = "https://nlp-ai-demo.example.workers.dev";

test("streams binary audio to the fixed service with path, query, credentials and Origin intact", async () => {
  const audio = new Uint8Array([0, 255, 128, 13, 10, 0, 42]);
  const request = new Request(`${origin}/api/transcribe?language=ko&name=a%2Fb`, {
    method: "POST",
    headers: {
      Authorization: "Basic demo-credential",
      Origin: origin,
      Host: "malicious.example",
      "Content-Type": "audio/wav",
    },
    body: audio,
  });
  const result = await worker.fetch(request, { DEMO_SERVICE: {
    async fetch(upstream) {
      assert.equal(upstream.url, "http://localhost:8765/api/transcribe?language=ko&name=a%2Fb");
      assert.equal(upstream.method, "POST");
      assert.equal(upstream.headers.get("Host"), "localhost:8765");
      assert.equal(upstream.headers.get("Authorization"), "Basic demo-credential");
      assert.equal(upstream.headers.get("Origin"), origin);
      assert.equal(upstream.headers.get("Content-Type"), "audio/wav");
      assert.equal(upstream.redirect, "manual");
      assert.equal(upstream.cache, "no-store");
      assert.ok(upstream.body instanceof ReadableStream);
      assert.deepEqual(new Uint8Array(await upstream.arrayBuffer()), audio);
      return new Response('{"text":"검사 결과 확인 뒤"}', {
        headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=3600" },
      });
    },
  } });
  assert.equal(result.status, 200);
  assert.equal(result.headers.get("Content-Type"), "application/json");
  assert.equal(result.headers.get("Cache-Control"), "no-store");
  assert.equal(result.headers.get("CDN-Cache-Control"), "no-store");
  assert.deepEqual(await result.json(), { text: "검사 결과 확인 뒤" });
});

for (const method of ["GET", "HEAD"]) {
  test(`${method} forwards without a request body`, async () => {
    const result = await worker.fetch(new Request(`${origin}/?next=https://external.example`, { method }), {
      DEMO_SERVICE: { async fetch(upstream) {
        assert.equal(upstream.url, "http://localhost:8765/?next=https://external.example");
        assert.equal(upstream.method, method);
        assert.equal(upstream.body, null);
        return new Response(method === "HEAD" ? null : "demo");
      } },
    });
    assert.equal(await result.text(), method === "HEAD" ? "" : "demo");
  });
}

test("does not follow or expose an upstream redirect", async () => {
  let calls = 0;
  const result = await worker.fetch(new Request(`${origin}/api/generate`, {
    method: "POST", headers: { Authorization: "Basic secret" }, body: "private transcript",
  }), { DEMO_SERVICE: { async fetch(upstream) {
    calls += 1;
    assert.equal(upstream.redirect, "manual");
    return new Response("upstream private detail", {
      status: 307, headers: { Location: "https://external.example/collect" },
    });
  } } });
  assert.equal(calls, 1);
  assert.equal(result.status, 503);
  assert.equal(result.headers.get("Location"), null);
  assert.doesNotMatch(await result.text(), /secret|private|external\.example/);
});

test("keeps the existing authentication challenge and disables caching", async () => {
  const result = await worker.fetch(new Request(origin), { DEMO_SERVICE: {
    async fetch() {
      return new Response("로그인이 필요합니다.", {
        status: 401, headers: { "WWW-Authenticate": 'Basic realm="NLP-AI Demo"' },
      });
    },
  } });
  assert.equal(result.status, 401);
  assert.equal(result.headers.get("WWW-Authenticate"), 'Basic realm="NLP-AI Demo"');
  assert.equal(result.headers.get("Cache-Control"), "no-store");
});

for (const method of ["GET", "HEAD"]) {
  test(`${method} returns a generic Korean 503 without disclosing failure details`, async () => {
    const result = await worker.fetch(new Request(`${origin}/?private=patient`, { method }), {
      DEMO_SERVICE: { async fetch() { throw new Error("Basic secret patient transcript"); } },
    });
    assert.equal(result.status, 503);
    assert.equal(result.headers.get("Retry-After"), "10");
    assert.equal(result.headers.get("Cache-Control"), "no-store");
    const text = await result.text();
    if (method === "HEAD") assert.equal(text, "");
    else assert.match(text, /데모 서버에 연결할 수 없습니다/);
    assert.doesNotMatch(text, /secret|patient|transcript/);
  });
}
