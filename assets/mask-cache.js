/*
 * IndexedDB cache of raw segmentation masks (the model's S×S uint8 output).
 *
 * Inference is the only expensive, deterministic step of background removal,
 * so its result is what gets cached; refinement re-runs from it in about a
 * second. Keys combine the decoded pixels' hash with the model, variant and
 * the exported graph's sha256, so a re-exported model never serves stale masks.
 *
 * Everything here is best effort: blocked IndexedDB, private browsing or a
 * full quota mean "no cache", never a failed run.
 */

const DB_NAME = 'drikus-upscale';
const STORE = 'masks';       // key -> { mask, size }
const META = 'mask-meta';    // key -> { bytes, used }: small, so LRU scans never read masks
const MAX_BYTES = 100 * 1024 * 1024;
const MAX_ENTRIES = 200;

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    let request;
    try {
      request = indexedDB.open(DB_NAME, 1);
    } catch {
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE, { keyPath: 'key' });
      request.result.createObjectStore(META, { keyPath: 'key' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
  return dbPromise;
}

const done = (tx) => new Promise((resolve, reject) => {
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error);
  tx.onabort = () => reject(tx.error);
});

const result = (request) => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

/** SHA-256 of the decoded RGBA pixels (not the file bytes), plus dimensions. */
export async function hashPixels(pixels, width, height) {
  const digest = await crypto.subtle.digest('SHA-256', pixels);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex}:${width}x${height}`;
}

export const maskKey = (pixelHash, modelId, variant) =>
  `${pixelHash}:${modelId}:${variant.precision}:${(variant.sha256 || 'dev').slice(0, 16)}`;

/** The cached mask for `key`, or null. Touches its LRU timestamp. */
export async function getMask(key) {
  try {
    const db = await openDb();
    if (!db) return null;
    const tx = db.transaction([STORE, META], 'readwrite');
    const entry = await result(tx.objectStore(STORE).get(key));
    if (entry) tx.objectStore(META).put({ key, bytes: entry.mask.byteLength, used: Date.now() });
    await done(tx);
    return entry ? { mask: new Uint8Array(entry.mask), size: entry.size } : null;
  } catch {
    return null;
  }
}

export async function putMask(key, mask, size) {
  try {
    const db = await openDb();
    if (!db) return;
    if (navigator.storage?.estimate) {
      const { usage = 0, quota = 0 } = await navigator.storage.estimate();
      if (quota && usage + mask.byteLength > quota * 0.9) return;
    }
    const tx = db.transaction([STORE, META], 'readwrite');
    tx.objectStore(STORE).put({ key, mask: mask.slice().buffer, size });
    tx.objectStore(META).put({ key, bytes: mask.byteLength, used: Date.now() });
    await done(tx);
    await evict(db);
  } catch {
    /* quota or a blocked store: skip caching */
  }
}

/** Drop least-recently-used masks beyond MAX_BYTES or MAX_ENTRIES. */
async function evict(db) {
  const tx = db.transaction([STORE, META], 'readwrite');
  const meta = await result(tx.objectStore(META).getAll());
  meta.sort((a, b) => b.used - a.used);               // newest first
  let bytes = 0;
  meta.forEach((entry, i) => {
    bytes += entry.bytes;
    if (i >= MAX_ENTRIES || bytes > MAX_BYTES) {
      tx.objectStore(STORE).delete(entry.key);
      tx.objectStore(META).delete(entry.key);
    }
  });
  await done(tx);
}

export async function clearMasks() {
  try {
    const db = await openDb();
    if (!db) return;
    const tx = db.transaction([STORE, META], 'readwrite');
    tx.objectStore(STORE).clear();
    tx.objectStore(META).clear();
    await done(tx);
  } catch { /* nothing to clear */ }
}
