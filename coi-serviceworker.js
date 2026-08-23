/*
 * GitHub Pages cannot set response headers, and WebAssembly threads need the
 * page to be cross-origin isolated. This script does double duty: loaded from
 * the page it registers itself as a service worker, and running as that worker
 * it stamps COOP/COEP onto every same-origin response. Everything this app
 * loads is same-origin, so nothing else is affected.
 *
 * If registration fails the app still works — onnxruntime-web simply falls back
 * to a single WASM thread (and WebGPU, when present, does not care either way).
 */
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', (event) => {
    const request = event.request;
    if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response;            // opaque, leave alone
          const headers = new Headers(response.headers);
          headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
          return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
        })
        .catch((err) => new Response(String(err), { status: 502 })),
    );
  });
} else {
  (() => {
    const RELOADS = 'coi-serviceworker-reloads';
    if (window.crossOriginIsolated) return;
    if (!window.isSecureContext || !navigator.serviceWorker) return;

    let reloads = 0;
    try { reloads = Number(sessionStorage.getItem(RELOADS) || 0); } catch { return; }
    if (reloads >= 2) return;                                     // never loop

    const src = document.currentScript && document.currentScript.src;
    if (!src) return;

    navigator.serviceWorker.register(src, { scope: './' }).then(async () => {
      await navigator.serviceWorker.ready;
      // This document was served before the worker existed, so it is not
      // isolated yet; one reload goes through the worker and picks up COOP/COEP.
      try { sessionStorage.setItem(RELOADS, String(reloads + 1)); } catch { /* ignore */ }
      window.location.reload();
    }).catch(() => { /* isolation is a bonus, not a requirement */ });
  })();
}
