/* Upscaling worker: owns onnxruntime-web, model caching and tiled inference. */
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';
import { modelById } from './models.js';

const MODEL_CACHE = 'drikus-upscale-models-v1';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.proxy = false;
ort.env.logLevel = 'error';

let cancelled = false;
const sessions = new Map();   // modelId -> { session, backend }
let backendReport = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const log = (text, kind) => post({ type: 'log', text, kind });

async function detectBackends() {
  if (backendReport) return backendReport;
  let webgpu = false;
  let adapter = null;
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      webgpu = !!adapter;
    } catch {
      webgpu = false;
    }
  }
  backendReport = {
    webgpu,
    adapter: adapter ? (adapter.info && adapter.info.description) || (adapter.info && adapter.info.vendor) || '' : '',
    crossOriginIsolated: !!self.crossOriginIsolated,
    threads: self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1,
    cores: navigator.hardwareConcurrency || 0,
  };
  return backendReport;
}

/* ---------- model bytes: network -> Cache API -> memory ---------- */

async function readModelBytes(model) {
  const url = new URL('../' + model.file, import.meta.url).href;
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
  } catch { /* private mode / insecure context: just fetch */ }

  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      post({ type: 'download', modelId: model.id, loaded: model.bytes, total: model.bytes, cached: true });
      return new Uint8Array(await hit.arrayBuffer());
    }
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download the model (HTTP ${res.status})`);
  if (cache) {
    try { await cache.put(url, res.clone()); } catch { /* quota — not fatal */ }
  }

  const total = Number(res.headers.get('content-length')) || model.bytes;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());

  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    post({ type: 'download', modelId: model.id, loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes;
}

async function getSession(modelId, preferred) {
  const cached = sessions.get(modelId);
  if (cached && (preferred === 'auto' || cached.backend === preferred)) return cached;

  const model = modelById(modelId);
  const caps = await detectBackends();
  const wanted = preferred === 'auto' ? (caps.webgpu ? 'webgpu' : 'wasm') : preferred;

  ort.env.wasm.numThreads = caps.threads;
  const bytes = await readModelBytes(model);

  const attempts = wanted === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError = null;
  for (const backend of attempts) {
    try {
      log(`Compiling ${model.name} for ${backend.toUpperCase()}…`);
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [backend],
        graphOptimizationLevel: 'all',
      });
      const entry = { session, backend, scale: model.scale };
      sessions.get(modelId)?.session?.release?.();
      sessions.set(modelId, entry);
      return entry;
    } catch (err) {
      lastError = err;
      if (backend === 'webgpu') log(`WebGPU could not run this model (${err.message}); falling back to WASM.`, 'warn');
    }
  }
  throw lastError || new Error('no execution provider could load the model');
}

/* ---------- inference ---------- */

async function warmup(entry, tile) {
  const size = Math.min(tile, 64);
  const input = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]);
  const out = await entry.session.run({ input });
  out.output?.dispose?.();
  input.dispose?.();
}

/** Pull an RGB tile out of the source RGBA buffer, clamping at the edges. */
function extractTile(src, srcW, srcH, x0, y0, w, h) {
  const data = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let y = 0; y < h; y++) {
    const sy = Math.min(srcH - 1, Math.max(0, y0 + y));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(srcW - 1, Math.max(0, x0 + x));
      const s = (sy * srcW + sx) * 4;
      const d = y * w + x;
      data[d] = src[s] / 255;
      data[plane + d] = src[s + 1] / 255;
      data[2 * plane + d] = src[s + 2] / 255;
    }
  }
  return data;
}

/** Crop the valid centre out of a model output plane and pack it as RGBA. */
function packTile(out, outW, outH, cropX, cropY, cropW, cropH) {
  const rgba = new Uint8ClampedArray(cropW * cropH * 4);
  const plane = outW * outH;
  for (let y = 0; y < cropH; y++) {
    const sy = cropY + y;
    for (let x = 0; x < cropW; x++) {
      const s = sy * outW + (cropX + x);
      const d = (y * cropW + x) * 4;
      rgba[d] = out[s] * 255;
      rgba[d + 1] = out[plane + s] * 255;
      rgba[d + 2] = out[2 * plane + s] * 255;
      rgba[d + 3] = 255;
    }
  }
  return rgba;
}

async function run(job) {
  cancelled = false;
  const { pixels, width, height, modelId, backend: preferred, tileSize, overlap } = job;
  const src = new Uint8ClampedArray(pixels);

  const entry = await getSession(modelId, preferred);
  const scale = entry.scale;
  post({ type: 'ready', backend: entry.backend, scale, caps: await detectBackends() });

  // Every tile is inferred at exactly the same tensor shape, edges included:
  // extractTile clamps out-of-bounds reads to the border pixel, so the model
  // always sees `tile + 2 * pad` square input. That keeps WebGPU from
  // recompiling shaders for the ragged right and bottom edges.
  const tile = Math.max(32, Math.min(tileSize, 1024, Math.max(width, height)));
  const pad = Math.max(0, Math.min(overlap, 64));
  const inW = tile + pad * 2;
  const inH = tile + pad * 2;
  const cols = Math.ceil(width / tile);
  const rows = Math.ceil(height / tile);
  const total = cols * rows;

  await warmup(entry, tile);
  if (cancelled) return post({ type: 'cancelled' });

  post({ type: 'start', total, outWidth: width * scale, outHeight: height * scale, backend: entry.backend });

  let done = 0;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (cancelled) return post({ type: 'cancelled' });

      const tx = col * tile;
      const ty = row * tile;
      const tw = Math.min(tile, width - tx);
      const th = Math.min(tile, height - ty);

      const data = extractTile(src, width, height, tx - pad, ty - pad, inW, inH);
      const input = new ort.Tensor('float32', data, [1, 3, inH, inW]);
      let result;
      try {
        result = await entry.session.run({ input });
      } finally {
        input.dispose?.();
      }
      const out = result.output.data;
      const outW = inW * scale;
      const outH = inH * scale;
      const rgba = packTile(out, outW, outH, pad * scale, pad * scale, tw * scale, th * scale);
      result.output.dispose?.();

      done++;
      post({
        type: 'tile',
        x: tx * scale, y: ty * scale, w: tw * scale, h: th * scale,
        pixels: rgba.buffer, done, total,
      }, [rgba.buffer]);

      // Yield so cancel messages get a chance to land between tiles.
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  post({ type: 'done' });
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === 'probe') {
      post({ type: 'caps', caps: await detectBackends() });
    } else if (msg.type === 'run') {
      await run(msg);
    } else if (msg.type === 'cancel') {
      cancelled = true;
    } else if (msg.type === 'evict') {
      for (const entry of sessions.values()) entry.session.release?.();
      sessions.clear();
      try { await caches.delete(MODEL_CACHE); } catch { /* ignore */ }
      post({ type: 'evicted' });
    }
  } catch (err) {
    post({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
