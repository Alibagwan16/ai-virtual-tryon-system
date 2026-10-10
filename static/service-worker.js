// Minimal service worker for the "AI Virtual Try-On Studio" PWA.
//
// This app is dynamic (Streamlit server-rendered + live AI generation), so
// there is no meaningful "offline mode" to build here — the only reason
// this file exists is that Chrome/Android's "Add to Home Screen" install
// prompt requires an active service worker with a fetch handler as one of
// its installability criteria. This one simply passes every request
// straight through to the network unchanged.
//
// If you later want real offline support (e.g. caching the app shell so a
// "you're offline" screen shows instead of a blank page), expand the
// fetch handler below with a cache-first/network-first strategy.

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});
