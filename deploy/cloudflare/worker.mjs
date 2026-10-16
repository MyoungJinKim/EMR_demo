function unavailable(method) {
  return new Response(method === "HEAD" ? null : "데모 서버에 연결할 수 없습니다. Mac의 실행 상태를 확인한 후 잠시 뒤 다시 시도해 주세요.", {
    status: 503,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
      "CDN-Cache-Control": "no-store",
      "Retry-After": "10",
    },
  });
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      const target = `http://localhost:8765${url.pathname}${url.search}`;
      const upstream = new Request(target, request);
      upstream.headers.set("Host", "localhost:8765");
      const response = await env.DEMO_SERVICE.fetch(new Request(upstream, {
        redirect: "manual",
        cache: "no-store",
      }));
      // Demo routes do not redirect; never forward credentials to another destination.
      if (response.status >= 300 && response.status < 400) {
        await response.body?.cancel();
        return unavailable(request.method);
      }
      const headers = new Headers(response.headers);
      headers.set("Cache-Control", "no-store");
      headers.set("CDN-Cache-Control", "no-store");
      return new Response(request.method === "HEAD" ? null : response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    } catch {
      return unavailable(request.method);
    }
  },
};
