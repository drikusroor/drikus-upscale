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

function releaseSession(modelId) {
  const entry = sessions.get(modelId);
  try { entry?.session?.release?.(); } catch { /* already torn down */ }
  sessions.delete(modelId);
}

async function createSession(model, backend) {
  log(`Compiling ${model.name} for ${backend.toUpperCase()}…`);
  const bytes = await readModelBytes(model);
  const options = { executionProviders: [backend], graphOptimizationLevel: 'all' };
  if (backend === 'webgpu') {
    // Belt and braces against "Shape mismatch attempting to re-use buffer":
    // the models no longer claim their output is input-sized (see
    // tools/export_onnx.py), but WebGPU's buffer planner is the strict one, so
    // it also gets the pattern planner switched off. WASM is left on its
    // defaults — measurably faster there, and it never had the problem.
    options.enableMemPattern = false;
  }
  return ort.InferenceSession.create(bytes, options);
}

/**
 * Get a session for one model at one fixed input shape.
 *
 * `shapeKey` pins the tile shape a session was built for, so changing the tile
 * size rebuilds rather than reusing a session whose buffer plan was laid out
 * for the old shape.
 */
async function getSession(modelId, preferred, shapeKey) {
  const cached = sessions.get(modelId);
  if (cached && cached.shapeKey === shapeKey && (preferred === 'auto' || cached.backend === preferred)) {
    return cached;
  }

  const model = modelById(modelId);
  const caps = await detectBackends();
  const wanted = preferred === 'auto' ? (caps.webgpu ? 'webgpu' : 'wasm') : preferred;
  ort.env.wasm.numThreads = caps.threads;

  const attempts = wanted === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError = null;
  for (const backend of attempts) {
    try {
      const session = await createSession(model, backend);
      releaseSession(modelId);
      const entry = { session, backend, shapeKey, scale: model.scale, modelId };
      sessions.set(modelId, entry);
      return entry;
    } catch (err) {
      lastError = err;
      if (backend === 'webgpu') log(`WebGPU could not load this model (${err.message}); falling back to WASM.`, 'warn');
    }
  }
  throw lastError || new Error('no execution provider could load the model');
}

/* ---------- inference ---------- */

const isBufferShapeError = (err) => /re-?use buffer|shape mismatch/i.test(err?.message || '');

/**
 * Run one tile, recovering from the WebGPU buffer-reuse failure mode: first by
 * rebuilding the session on the same backend, then by dropping to WASM for the
 * rest of the job rather than leaving the user with a dead end.
 */
async function infer(job, data, shape) {
  for (let attempt = 0; ; attempt++) {
    const input = new ort.Tensor('float32', data, shape);
    try {
      return await job.entry.session.run({ input });
    } catch (err) {
      const shapeError = isBufferShapeError(err);
      // A buffer-shape error is worth one rebuild on the same backend; anything
      // else that WebGPU throws goes straight to WASM. Either way the user ends
      // up with a finished image instead of a dead end.
      if (attempt >= 2 || !(shapeError || job.entry.backend === 'webgpu')) throw err;
      const backend = attempt === 0 && shapeError ? job.entry.backend : 'wasm';
      log(backend === job.entry.backend
        ? `Rebuilding the ${backend.toUpperCase()} session after a buffer-shape error…`
        : `WebGPU could not run this model (${err.message}); finishing on WASM.`, 'warn');
      releaseSession(job.modelId);
      const model = modelById(job.modelId);
      const session = await createSession(model, backend);
      job.entry = { session, backend, shapeKey: job.shapeKey, scale: model.scale, modelId: job.modelId };
      sessions.set(job.modelId, job.entry);
      post({ type: 'ready', backend, scale: model.scale, caps: await detectBackends() });
    } finally {
      input.dispose?.();
    }
  }
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

async function run(request) {
  cancelled = false;
  const { pixels, width, height, modelId, backend: preferred, tileSize, overlap } = request;
  const src = new Uint8ClampedArray(pixels);

  // Every tile is inferred at exactly the same tensor shape, edges included:
  // extractTile clamps out-of-bounds reads to the border pixel, so the model
  // always sees `tile + 2 * pad` square input. That keeps WebGPU from
  // recompiling shaders for the ragged right and bottom edges.
  const tile = Math.max(32, Math.min(tileSize, 1024, Math.max(width, height)));
  const pad = Math.max(0, Math.min(overlap, 64));
  const inW = tile + pad * 2;
  const inH = tile + pad * 2;
  const shape = [1, 3, inH, inW];
  const shapeKey = `${inW}x${inH}`;
  const cols = Math.ceil(width / tile);
  const rows = Math.ceil(height / tile);
  const total = cols * rows;

  const job = { modelId, shapeKey, entry: await getSession(modelId, preferred, shapeKey) };
  const scale = job.entry.scale;
  post({ type: 'ready', backend: job.entry.backend, scale, caps: await detectBackends() });

  // WebGPU compiles its compute pipelines lazily, so a throwaway pass at the
  // exact tile shape removes a stall from the first real tile. It has to be the
  // exact shape: a differently shaped warmup is itself the kind of shape change
  // that upsets WebGPU's buffer planner. WASM has nothing to precompile and
  // measured slightly faster without it, so it skips straight to the tiles.
  if (job.entry.backend === 'webgpu') {
    const warm = await infer(job, new Float32Array(3 * inW * inH), shape);
    warm.output?.dispose?.();
  }
  if (cancelled) return post({ type: 'cancelled' });

  post({ type: 'start', total, outWidth: width * scale, outHeight: height * scale, backend: job.entry.backend });

  let done = 0;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (cancelled) return post({ type: 'cancelled' });

      const tx = col * tile;
      const ty = row * tile;
      const tw = Math.min(tile, width - tx);
      const th = Math.min(tile, height - ty);

      const data = extractTile(src, width, height, tx - pad, ty - pad, inW, inH);
      const result = await infer(job, data, shape);
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
