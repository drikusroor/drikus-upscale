/*
 * Inference worker: owns onnxruntime-web, model caching, and every tool's
 * heavy lifting — tiled upscaling and background removal — so all tools share
 * one ORT instance, one session map and one model cache.
 */
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';
import { modelById, pickVariant, variantsOf } from './models.js';
import { processStrip, Scratch, stripMargin, stripRows, DEFAULT_REFINE } from './matting.js';
import { hashPixels, maskKey, getMask, putMask, clearMasks } from './mask-cache.js';

const MODEL_CACHE = 'drikus-upscale-models-v1';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.proxy = false;
ort.env.logLevel = 'error';

let cancelled = false;
const sessions = new Map();   // modelId -> { session, backend, shapeKey, scale, modelId, variant }
let backendReport = null;
let inferences = 0;           // session.run calls; reported so tests can prove cache hits and refines skip the model

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const log = (text, kind) => post({ type: 'log', text, kind });
const stage = (name, done, total) => post({ type: 'stage', name, done, total });
const yieldToEvents = () => new Promise((r) => setTimeout(r, 0));

class Cancelled extends Error {}
const checkCancelled = () => { if (cancelled) throw new Cancelled(); };

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
    // Decides whether WebGPU gets the half-size fp16 segmentation weights.
    shaderF16: !!adapter && adapter.features.has('shader-f16'),
    crossOriginIsolated: !!self.crossOriginIsolated,
    threads: self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1,
    cores: navigator.hardwareConcurrency || 0,
  };
  return backendReport;
}

/* ---------- model bytes: network -> Cache API -> memory ---------- */

const fileUrl = (file) => new URL('../' + file, import.meta.url).href;

async function openModelCache() {
  try { return await caches.open(MODEL_CACHE); } catch { return null; }   // private mode / insecure context
}

/** Whether every file of a variant is already in the Cache API. */
async function isCached(variant) {
  const cache = await openModelCache();
  if (!cache) return false;
  for (const file of variant.files) {
    if (!(await cache.match(fileUrl(file)))) return false;
  }
  return true;
}

/**
 * Read every file of a variant — the graph, then any external-data shards —
 * each cached separately, with download progress summed over all of them.
 */
async function readVariantFiles(model, variant) {
  const cache = await openModelCache();
  const total = variant.bytes || variantsOf(model)[0].bytes || 0;
  let before = 0;
  const out = [];
  for (const file of variant.files) {
    const url = fileUrl(file);
    const hit = cache && await cache.match(url);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      before += bytes.byteLength;
      post({ type: 'download', modelId: model.id, loaded: before, total: Math.max(total, before), cached: true });
      out.push(bytes);
      continue;
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error(`could not download the model (HTTP ${res.status})`);
    // Store in the background while the body streams in for progress; awaiting
    // the put first would drain the whole download before any progress shows.
    const storing = cache ? cache.put(url, res.clone()).catch(() => { /* quota — not fatal */ }) : null;
    let bytes;
    if (!res.body) {
      bytes = new Uint8Array(await res.arrayBuffer());
    } else {
      const reader = res.body.getReader();
      const chunks = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        post({ type: 'download', modelId: model.id, loaded: before + loaded, total: Math.max(total, before + loaded) });
      }
      bytes = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    }
    before += bytes.byteLength;
    if (storing) await storing;
    out.push(bytes);
    checkCancelled();
  }
  return out;
}

function releaseSession(modelId) {
  const entry = sessions.get(modelId);
  try { entry?.session?.release?.(); } catch { /* already torn down */ }
  sessions.delete(modelId);
}

async function createSession(model, backend) {
  const caps = await detectBackends();
  const variant = pickVariant(model, backend, caps);
  // A 1024 px segmentation network and an upscaler together can exceed the
  // 4 GB wasm32 heap, so on WASM only one tool's sessions stay resident.
  // WebGPU keeps its weights in GPU memory and keeps everything.
  if (backend === 'wasm') {
    for (const entry of [...sessions.values()]) {
      if (modelById(entry.modelId).task !== model.task) releaseSession(entry.modelId);
    }
  }
  const files = await readVariantFiles(model, variant);
  checkCancelled();
  log(`Compiling ${model.name}${variant.precision === 'fp16' ? ' (fp16)' : ''} for ${backend.toUpperCase()}…`);
  stage('compile');
  const options = { executionProviders: [backend], graphOptimizationLevel: 'all' };
  if (backend === 'webgpu') {
    // Belt and braces against "Shape mismatch attempting to re-use buffer":
    // the models no longer claim their output is input-sized (see
    // tools/export_onnx.py), but WebGPU's buffer planner is the strict one, so
    // it also gets the pattern planner switched off. WASM is left on its
    // defaults — measurably faster there, and it never had the problem.
    options.enableMemPattern = false;
  }
  if (files.length > 1) {
    options.externalData = files.slice(1).map((data, i) => ({
      path: variant.files[i + 1].split('/').pop(),
      data,
    }));
  }
  const session = await ort.InferenceSession.create(files[0], options);
  return { session, variant };
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
      const { session, variant } = await createSession(model, backend);
      releaseSession(modelId);
      const entry = { session, backend, shapeKey, scale: model.scale, modelId, variant };
      sessions.set(modelId, entry);
      return entry;
    } catch (err) {
      if (err instanceof Cancelled) throw err;
      lastError = err;
      if (backend === 'webgpu') log(`WebGPU could not load this model (${err.message}); falling back to WASM.`, 'warn');
    }
  }
  throw lastError || new Error('no execution provider could load the model');
}

/* ---------- inference ---------- */

const isBufferShapeError = (err) => /re-?use buffer|shape mismatch/i.test(err?.message || '');

/**
 * Run one tensor, recovering from the WebGPU buffer-reuse failure mode: first
 * by rebuilding the session on the same backend, then by dropping to WASM for
 * the rest of the job rather than leaving the user with a dead end. Dropping
 * to WASM also swaps the weights to the variant WASM runs (fp16 → fp32).
 */
async function infer(job, data, shape) {
  for (let attempt = 0; ; attempt++) {
    const input = new ort.Tensor('float32', data, shape);
    try {
      const result = await job.entry.session.run({ input });
      inferences++;
      return result;
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
      const { session, variant } = await createSession(model, backend);
      job.entry = { session, backend, shapeKey: job.shapeKey, scale: model.scale, modelId: job.modelId, variant };
      sessions.set(job.modelId, job.entry);
      post({ type: 'ready', backend, scale: model.scale, caps: await detectBackends() });
    } finally {
      input.dispose?.();
    }
  }
}

/* ---------- upscaling ---------- */

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

async function upscale(request) {
  const { pixels, width, height, modelId, backend: preferred, tileSize, overlap } = request;
  const src = new Uint8ClampedArray(pixels);

  // Every tile is inferred at exactly the same tensor shape, edges included:
  // extractTile clamps out-of-bounds reads to the border pixel, so the model
  // always sees `tile + 2 * pad` square input. That keeps WebGPU from
  // recompiling shaders for the ragged right and bottom edges.
  //
  // A model with fixedTile/fixedContextPad (Swin2SR) was exported at one exact
  // input resolution -- its attention mask is baked in as a constant for that
  // shape at trace time (see tools/swin2sr/export.py), so any other shape would
  // silently produce a wrong mask rather than fail loudly. The tile/overlap
  // sliders are locked in the UI for such a model, but the worker enforces it
  // independently of what the request happens to carry.
  const fixedModel = modelById(modelId);
  const tile = fixedModel.fixedTile
    ? fixedModel.fixedTile
    : Math.max(32, Math.min(tileSize, 1024, Math.max(width, height)));
  const pad = fixedModel.fixedTile
    ? fixedModel.fixedContextPad
    : Math.max(0, Math.min(overlap, 64));
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
  checkCancelled();

  post({ type: 'start', total, outWidth: width * scale, outHeight: height * scale, backend: job.entry.backend });

  let done = 0;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      checkCancelled();

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
      await yieldToEvents();
    }
  }
  post({ type: 'done', inferences });
}

/* ---------- background removal ---------- */

// The image being cut out, kept so a refine-slider change re-runs only the
// post-processing (FR-7). This is the worker's one full-resolution copy;
// output strips are transferred away as soon as they are packed.
let held = null;            // { src, width, height, mask, size }
let refineGeneration = 0;
const scratch = new Scratch();

/**
 * Resize the whole image to S×S as the model's float32 [1, 3, S, S] input —
 * squashed rather than letterboxed, matching how both models were trained.
 * No tiling: segmentation needs global context, unlike upscaling.
 */
function segmentationInput(src, width, height, size) {
  const data = new Float32Array(3 * size * size);
  const plane = size * size;
  const fx = width / size;
  const fy = height / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = y * size + x;
      let r = 0, g = 0, b = 0;
      if (fx <= 1 && fy <= 1) {
        // Enlarging: bilinear between pixel centres.
        const sx = Math.min(width - 1, Math.max(0, (x + 0.5) * fx - 0.5));
        const sy = Math.min(height - 1, Math.max(0, (y + 0.5) * fy - 0.5));
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
        const wx = sx - x0, wy = sy - y0;
        const p00 = (y0 * width + x0) * 4, p01 = (y0 * width + x1) * 4;
        const p10 = (y1 * width + x0) * 4, p11 = (y1 * width + x1) * 4;
        const w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy), w10 = (1 - wx) * wy, w11 = wx * wy;
        r = src[p00] * w00 + src[p01] * w01 + src[p10] * w10 + src[p11] * w11;
        g = src[p00 + 1] * w00 + src[p01 + 1] * w01 + src[p10 + 1] * w10 + src[p11 + 1] * w11;
        b = src[p00 + 2] * w00 + src[p01 + 2] * w01 + src[p10 + 2] * w10 + src[p11 + 2] * w11;
      } else {
        // Shrinking (on either axis): average the source footprint, so fine
        // texture doesn't alias into the model's view.
        const xa = Math.floor(x * fx), xb = Math.max(xa + 1, Math.min(width, Math.floor((x + 1) * fx)));
        const ya = Math.floor(y * fy), yb = Math.max(ya + 1, Math.min(height, Math.floor((y + 1) * fy)));
        for (let sy = ya; sy < yb; sy++) {
          for (let sx = xa, s = (sy * width + xa) * 4; sx < xb; sx++, s += 4) {
            r += src[s]; g += src[s + 1]; b += src[s + 2];
          }
        }
        const n = (xb - xa) * (yb - ya);
        r /= n; g /= n; b /= n;
      }
      data[d] = r / 255;
      data[plane + d] = g / 255;
      data[2 * plane + d] = b / 255;
    }
  }
  return data;
}

/** Run one segmentation model on the held image; returns its S×S uint8 mask. */
async function segment(model, preferred, pixelHash) {
  const size = model.inputSize;
  const caps = await detectBackends();
  const wanted = preferred === 'auto' ? (caps.webgpu ? 'webgpu' : 'wasm') : preferred;
  const variant = pickVariant(model, wanted, caps);
  const key = maskKey(pixelHash, model.id, variant);
  const hit = await getMask(key);
  if (hit && hit.size === size) return { mask: hit.mask, cached: true, backend: wanted };

  const shapeKey = `${size}x${size}`;
  const job = { modelId: model.id, shapeKey, entry: await getSession(model.id, preferred, shapeKey) };
  post({ type: 'ready', backend: job.entry.backend, scale: 1, caps });
  checkCancelled();
  // No warm-up pass here, unlike upscaling: a segmentation model runs once
  // per image, so a throwaway run would double the cost instead of hiding
  // a stall between tiles.
  stage('infer', 0, 1);
  const t0 = performance.now();
  const result = await infer(job, segmentationInput(held.src, held.width, held.height, size), [1, 3, size, size]);
  const seconds = (performance.now() - t0) / 1000;
  const alpha = result.output.data;
  const mask = new Uint8Array(size * size);
  for (let i = 0; i < mask.length; i++) mask[i] = alpha[i] * 255 + 0.5;
  result.output.dispose?.();
  // The variant that actually ran — a WebGPU failure may have swapped it.
  await putMask(maskKey(pixelHash, model.id, job.entry.variant), mask, size);
  return { mask, cached: false, backend: job.entry.backend, seconds };
}

async function remove(request) {
  const { pixels, width, height, modelId, backend: preferred, refine } = request;
  const src = new Uint8ClampedArray(pixels);
  held = { src, width, height, mask: null, size: 0 };
  const model = modelById(modelId, 'remove');

  stage('hash');
  const pixelHash = await hashPixels(src, width, height);
  checkCancelled();

  // Instant preview: if the fast model's weights are already on the device,
  // run it first and show its mask while the main model downloads and runs —
  // unless the main model's mask is cached, in which case it is instant anyway.
  const caps = await detectBackends();
  const wanted = preferred === 'auto' ? (caps.webgpu ? 'webgpu' : 'wasm') : preferred;
  const previewModel = model.preview && modelById(model.preview, 'remove');
  if (previewModel && !(await getMask(maskKey(pixelHash, model.id, pickVariant(model, wanted, caps))))) {
    if (await isCached(pickVariant(previewModel, wanted, caps))) {
      try {
        const preview = await segment(previewModel, preferred, pixelHash);
        post({ type: 'mask', preview: true, size: previewModel.inputSize, mask: preview.mask.buffer }, [preview.mask.buffer]);
      } catch (err) {
        if (err instanceof Cancelled) throw err;
        /* a preview is a nicety; the real run continues */
      }
      checkCancelled();
    }
  }

  const result = await segment(model, preferred, pixelHash);
  checkCancelled();
  held.mask = result.mask;
  held.size = model.inputSize;
  const copy = result.mask.slice();
  post({
    type: 'mask', preview: false, size: model.inputSize, mask: copy.buffer,
    cached: result.cached, backend: result.backend, seconds: result.seconds,
  }, [copy.buffer]);

  await postProcess(refine, ++refineGeneration);
}

/**
 * Upsample, refine, decontaminate and stream the cutout in strips. A newer
 * refine request (or a cancel) abandons this one between strips.
 */
async function postProcess(refine, generation) {
  const { src, width, height, mask, size } = held;
  const params = { ...DEFAULT_REFINE, ...refine };
  const rows = stripRows(width, stripMargin(params));
  const total = Math.ceil(height / rows);
  const t0 = performance.now();
  post({ type: 'cutout-start', width, height, total, generation });
  for (let i = 0; i < total; i++) {
    if (generation !== refineGeneration) return;
    checkCancelled();
    stage(params.decontaminate ? 'decontaminate' : 'refine', i, total);
    const y0 = i * rows;
    const y1 = Math.min(height, y0 + rows);
    const rgba = processStrip({ src, width, height, mask, size, refine: params, y0, y1, scratch });
    post({ type: 'cutout', y: y0, h: y1 - y0, width, pixels: rgba.buffer, done: i + 1, total, generation }, [rgba.buffer]);
    await yieldToEvents();
  }
  if (generation !== refineGeneration) return;
  post({ type: 'done', generation, refineSeconds: (performance.now() - t0) / 1000, inferences });
}

/* ---------- messages ---------- */

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === 'probe') {
      post({ type: 'caps', caps: await detectBackends() });
    } else if (msg.type === 'run') {
      cancelled = false;
      await upscale(msg);
    } else if (msg.type === 'remove') {
      cancelled = false;
      refineGeneration++;
      await remove(msg);
    } else if (msg.type === 'refine') {
      if (!held?.mask) return;
      cancelled = false;
      await postProcess(msg.refine, ++refineGeneration);
    } else if (msg.type === 'release-image') {
      held = null;
      refineGeneration++;
      scratch.release();
    } else if (msg.type === 'cancel') {
      cancelled = true;
    } else if (msg.type === 'evict') {
      for (const entry of sessions.values()) entry.session.release?.();
      sessions.clear();
      try { await caches.delete(MODEL_CACHE); } catch { /* ignore */ }
      await clearMasks();
      post({ type: 'evicted' });
    }
  } catch (err) {
    if (err instanceof Cancelled) {
      post({ type: 'cancelled' });
      return;
    }
    post({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
