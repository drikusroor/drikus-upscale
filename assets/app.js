import { MODELS, DEFAULT_MODEL_ID, modelById, formatBytes } from './models.js';

const $ = (id) => document.getElementById(id);
const MAX_LONG_SIDE = 4096;        // 4× of this is 16384 px — the browser canvas ceiling
const STORE_KEY = 'drikus-upscale/settings';
const CALIB_KEY = 'drikus-upscale/calibration';

const el = {
  dropzone: $('dropzone'), fileInput: $('file-input'), browse: $('browse'),
  workspace: $('workspace'), thumb: $('thumb'), srcName: $('src-name'), srcDims: $('src-dims'),
  changeImage: $('change-image'), modelSelect: $('model-select'), modelBlurb: $('model-blurb'), modelMeta: $('model-meta'),
  modelEstimate: $('model-estimate'), scaleSelect: $('scale-select'), backendSelect: $('backend-select'),
  tileSize: $('tile-size'), tileValue: $('tile-value'), overlap: $('overlap'), overlapValue: $('overlap-value'),
  clearCache: $('clear-cache'), run: $('run'), cancel: $('cancel'), progress: $('progress'),
  barFill: $('bar-fill'), status: $('status'), error: $('error'), capsChip: $('caps-chip'),
  before: $('canvas-before'), after: $('canvas-after'), clip: $('clip'), handle: $('handle'),
  compare: $('compare'), stage: $('stage'), tabCompare: $('tab-compare'), tabResult: $('tab-result'),
  downloadPng: $('download-png'), downloadWebp: $('download-webp'), outDims: $('out-dims'),
  previewHint: $('preview-hint'), pasteKey: $('paste-key'),
};

const state = {
  worker: null,
  source: null,          // { bitmap, width, height, name, imageData, hasAlpha }
  alphaCtx: null,        // upscaled alpha, only when the source has transparency
  running: false,
  startedAt: 0,
  modelScale: 4,
  settings: loadSettings(),
  calibration: loadCalibration(),
  backendUsed: 'wasm',
};

function loadCalibration() {
  try { return JSON.parse(localStorage.getItem(CALIB_KEY) || '{}'); } catch { return {}; }
}

/* --------------------------------------------------------------- settings */

function loadSettings() {
  const defaults = { model: DEFAULT_MODEL_ID, scale: '2', backend: 'auto', tile: 192, overlap: 16 };
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(STORE_KEY) || '{}') };
  } catch {
    return defaults;
  }
}

function saveSettings() {
  state.settings = {
    model: el.modelSelect.value,
    scale: el.scaleSelect.value,
    backend: el.backendSelect.value,
    tile: Number(el.tileSize.value),
    overlap: Number(el.overlap.value),
  };
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state.settings)); } catch { /* private mode */ }
}

/* --------------------------------------------------------------- worker */

function getWorker() {
  if (state.worker) return state.worker;
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => onWorkerMessage(e.data);
  worker.onerror = (e) => {
    showError(`The upscaling worker crashed: ${e.message || 'unknown error'}`);
    finishRun();
    state.worker?.terminate();
    state.worker = null;
  };
  state.worker = worker;
  return worker;
}

/* --------------------------------------------------------------- UI setup */

function buildModelSelect() {
  el.modelSelect.innerHTML = '';
  for (const model of MODELS) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.recommended ? `${model.name}  ★` : model.name;
    el.modelSelect.append(option);
  }
  el.modelSelect.value = modelById(state.settings.model).id;
}

function describeModel() {
  const model = modelById(el.modelSelect.value);
  el.modelMeta.innerHTML = '';
  if (model.recommended) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'Recommended';
    el.modelMeta.append(badge);
  }
  el.modelMeta.append(document.createTextNode(`${model.scale}× · ${formatBytes(model.bytes)}`));
  el.modelBlurb.textContent = model.blurb;
  updateEstimate();
}

/** Which backend a run would actually use, given the picker and the hardware. */
function effectiveBackend() {
  const choice = el.backendSelect.value;
  if (choice === 'wasm') return 'wasm';
  if (choice === 'webgpu') return 'webgpu';
  return state.caps?.webgpu ? 'webgpu' : 'wasm';
}

/**
 * Throughput in "cost-weighted megapixels per second". The seeds are rough
 * measurements; every completed run replaces them with what this machine
 * actually did, so the estimate sharpens after the first upscale.
 */
function throughput(backend) {
  const calibrated = state.calibration[backend];
  if (calibrated) return calibrated;
  if (backend === 'webgpu') return 0.4;
  return 0.007 * Math.max(1, state.caps?.threads || 1);
}

function updateEstimate() {
  if (!state.source) { el.modelEstimate.textContent = ''; return; }
  const model = modelById(el.modelSelect.value);
  const mpx = (state.source.width * state.source.height) / 1e6;
  const backend = effectiveBackend();
  const seconds = (mpx * model.cost) / throughput(backend);
  const pretty = seconds < 2 ? 'a second or two'
    : seconds < 90 ? `about ${Math.round(seconds)} s`
      : `about ${Math.round(seconds / 60)} min`;
  const measured = state.calibration[backend] ? '' : ' (rough guess until the first run)';
  el.modelEstimate.textContent = `${mpx.toFixed(2)} Mpx input · ${formatBytes(model.bytes)} download on first use · ${pretty} on ${backend.toUpperCase()}${measured}.`;
  el.modelEstimate.classList.toggle('slow', seconds > 60);
}

function recordCalibration(backend, seconds) {
  const model = modelById(el.modelSelect.value);
  const mpx = (state.source.width * state.source.height) / 1e6;
  if (seconds < 0.4) return;                       // too short to measure anything
  const observed = (mpx * model.cost) / seconds;
  const previous = state.calibration[backend];
  state.calibration[backend] = previous ? previous * 0.4 + observed * 0.6 : observed;
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(state.calibration)); } catch { /* ignore */ }
}

function updateCapsChip(caps) {
  state.caps = caps;
  if (caps.webgpu) {
    el.capsChip.textContent = 'WebGPU';
    el.capsChip.className = 'chip chip-ok';
    el.capsChip.title = caps.adapter ? `WebGPU via ${caps.adapter}` : 'WebGPU available';
  } else {
    el.capsChip.textContent = `WASM · ${caps.threads} thread${caps.threads === 1 ? '' : 's'}`;
    el.capsChip.className = 'chip chip-muted';
    el.capsChip.title = caps.crossOriginIsolated
      ? 'No WebGPU adapter — running on multi-threaded WebAssembly'
      : 'No WebGPU adapter, and the page is not cross-origin isolated, so WASM runs single-threaded';
  }
  updateEstimate();
}

/* --------------------------------------------------------------- input */

async function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('That does not look like an image file.');
    return;
  }
  hideError();
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (err) {
    showError(`The browser could not decode that image (${err.message}).`);
    return;
  }
  if (Math.max(bitmap.width, bitmap.height) > MAX_LONG_SIDE) {
    showError(`That image is ${bitmap.width}×${bitmap.height}. The long side has to stay under ${MAX_LONG_SIDE} px so the 4× result fits inside a browser canvas — downscale it first, or crop the region you care about.`);
    bitmap.close();
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let hasAlpha = false;
  for (let i = 3; i < imageData.data.length; i += 4) {
    if (imageData.data[i] !== 255) { hasAlpha = true; break; }
  }

  state.source = { bitmap, width: bitmap.width, height: bitmap.height, name: file.name || 'pasted-image.png', imageData, hasAlpha };
  state.alphaCtx = null;

  el.srcName.textContent = state.source.name;
  el.srcDims.textContent = `${bitmap.width} × ${bitmap.height}${hasAlpha ? ' · has transparency' : ''}`;
  drawThumb(bitmap);
  el.dropzone.hidden = true;
  el.workspace.hidden = false;
  resetPreview();
  updateEstimate();
  el.run.focus();
}

function drawThumb(bitmap) {
  const ctx = el.thumb.getContext('2d');
  const size = el.thumb.width;
  const s = Math.max(size / bitmap.width, size / bitmap.height);
  const w = bitmap.width * s;
  const h = bitmap.height * s;
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(bitmap, (size - w) / 2, (size - h) / 2, w, h);
}

async function loadFromUrl(url) {
  try {
    const res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    await loadFile(new File([blob], url.split('/').pop() || 'pasted-image', { type: blob.type }));
  } catch (err) {
    showError(`Could not fetch that URL (${err.message}). Most sites block cross-origin reads — save the image and drop the file in instead.`);
  }
}

/* --------------------------------------------------------------- preview */

function resetPreview() {
  const { width, height } = state.source;
  sizeCanvases(width, height);
  const bctx = el.before.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'high';
  bctx.clearRect(0, 0, width, height);
  bctx.drawImage(state.source.bitmap, 0, 0, width, height);
  el.after.getContext('2d').clearRect(0, 0, width, height);
  el.outDims.textContent = '';
  el.downloadPng.disabled = true;
  el.downloadWebp.disabled = true;
  setCompareMode('compare');
  setSplit(0.5);
}

function sizeCanvases(width, height) {
  for (const canvas of [el.before, el.after]) {
    canvas.width = width;
    canvas.height = height;
  }
}

function setSplit(fraction) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  el.clip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  el.handle.style.left = `${pct}%`;
  el.handle.setAttribute('aria-valuenow', String(Math.round(pct)));
}

function setCompareMode(mode) {
  const resultOnly = mode === 'result';
  el.compare.classList.toggle('result-only', resultOnly);
  el.tabResult.classList.toggle('active', resultOnly);
  el.tabCompare.classList.toggle('active', !resultOnly);
}

/* --------------------------------------------------------------- run */

async function startRun() {
  if (!state.source || state.running) return;
  hideError();
  saveSettings();

  const model = modelById(el.modelSelect.value);
  state.modelScale = model.scale;
  const { width, height } = state.source;

  sizeCanvases(width * model.scale, height * model.scale);
  const bctx = el.before.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'high';
  bctx.drawImage(state.source.bitmap, 0, 0, el.before.width, el.before.height);
  el.after.getContext('2d').clearRect(0, 0, el.after.width, el.after.height);
  el.downloadPng.disabled = true;
  el.downloadWebp.disabled = true;

  if (state.source.hasAlpha) prepareAlpha(width * model.scale, height * model.scale);

  state.running = true;
  state.startedAt = performance.now();
  el.run.disabled = true;
  el.cancel.hidden = false;
  el.progress.hidden = false;
  el.barFill.style.width = '0%';
  setStatus(`Loading ${model.name}…`);

  // The worker consumes the pixel buffer, so hand it a copy we can throw away.
  const pixels = state.source.imageData.data.slice().buffer;
  getWorker().postMessage({
    type: 'run',
    pixels,
    width,
    height,
    modelId: model.id,
    backend: el.backendSelect.value,
    tileSize: Number(el.tileSize.value),
    overlap: Number(el.overlap.value),
  }, [pixels]);
}

function prepareAlpha(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(state.source.bitmap, 0, 0, width, height);
  state.alphaCtx = ctx;
}

function onWorkerMessage(msg) {
  switch (msg.type) {
    case 'caps':
      updateCapsChip(msg.caps);
      break;
    case 'log':
      setStatus(msg.text, msg.kind);
      break;
    case 'download': {
      const pct = msg.total ? Math.round((msg.loaded / msg.total) * 100) : 0;
      el.barFill.style.width = `${pct}%`;
      setStatus(msg.cached ? 'Model loaded from cache.' : `Downloading model… ${pct}% (${formatBytes(msg.loaded)} of ${formatBytes(msg.total)})`);
      break;
    }
    case 'ready':
      updateCapsChip(msg.caps);
      break;
    case 'start':
      state.startedAt = performance.now();
      state.backendUsed = msg.backend;
      setStatus(`Upscaling on ${msg.backend.toUpperCase()} — tile 0 of ${msg.total}…`);
      break;
    case 'tile':
      paintTile(msg);
      break;
    case 'done':
      completeRun();
      break;
    case 'cancelled':
      setStatus('Cancelled.');
      finishRun();
      break;
    case 'evicted':
      setStatus('Cached model downloads cleared.');
      break;
    case 'error':
      showError(msg.message);
      finishRun();
      break;
  }
}

function paintTile(msg) {
  const rgba = new Uint8ClampedArray(msg.pixels);
  if (state.alphaCtx) {
    const alpha = state.alphaCtx.getImageData(msg.x, msg.y, msg.w, msg.h).data;
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = alpha[i];
  }
  el.after.getContext('2d').putImageData(new ImageData(rgba, msg.w, msg.h), msg.x, msg.y);

  const pct = Math.round((msg.done / msg.total) * 100);
  el.barFill.style.width = `${pct}%`;
  const elapsed = (performance.now() - state.startedAt) / 1000;
  const remaining = (elapsed / msg.done) * (msg.total - msg.done);
  setStatus(`Tile ${msg.done} of ${msg.total} · ${elapsed.toFixed(1)} s elapsed · ~${remaining.toFixed(0)} s left`);
}

function completeRun() {
  const target = Number(el.scaleSelect.value);
  if (target !== state.modelScale) resampleTo(target);
  el.outDims.textContent = `${el.after.width} × ${el.after.height}`;
  el.downloadPng.disabled = false;
  el.downloadWebp.disabled = false;
  const elapsed = (performance.now() - state.startedAt) / 1000;
  recordCalibration(state.backendUsed, elapsed);
  setStatus(`Done in ${elapsed.toFixed(1)} s on ${state.backendUsed.toUpperCase()}.`);
  updateEstimate();
  finishRun();
}

/** Resample the native 4× result down to the requested output scale. */
function resampleTo(target) {
  const width = Math.round(state.source.width * target);
  const height = Math.round(state.source.height * target);
  const tmp = document.createElement('canvas');
  tmp.width = width;
  tmp.height = height;
  const tctx = tmp.getContext('2d');
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(el.after, 0, 0, width, height);

  sizeCanvases(width, height);
  const bctx = el.before.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'high';
  bctx.drawImage(state.source.bitmap, 0, 0, width, height);
  el.after.getContext('2d').drawImage(tmp, 0, 0);
}

function finishRun() {
  state.running = false;
  el.run.disabled = false;
  el.cancel.hidden = true;
}

/* --------------------------------------------------------------- helpers */

function setStatus(text, kind) {
  el.status.textContent = text;
  el.status.className = `status${kind === 'warn' ? ' warn' : ''}`;
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = false;
}

function hideError() { el.error.hidden = true; }

function download(type, extension, quality) {
  el.after.toBlob((blob) => {
    if (!blob) { showError('The browser refused to encode the result.'); return; }
    const base = state.source.name.replace(/\.[^.]+$/, '') || 'image';
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${base}_upscaled_${el.scaleSelect.value}x.${extension}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  }, type, quality);
}

/* --------------------------------------------------------------- events */

el.browse.addEventListener('click', (e) => { e.stopPropagation(); el.fileInput.click(); });
el.dropzone.addEventListener('click', () => el.fileInput.click());
el.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
});
el.changeImage.addEventListener('click', () => el.fileInput.click());
el.fileInput.addEventListener('change', () => {
  if (el.fileInput.files[0]) loadFile(el.fileInput.files[0]);
  el.fileInput.value = '';
});

let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  if (![...e.dataTransfer.types].includes('Files')) return;
  e.preventDefault();
  dragDepth++;
  el.dropzone.classList.add('dragging');
});
window.addEventListener('dragover', (e) => {
  if ([...e.dataTransfer.types].includes('Files')) e.preventDefault();
});
window.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) { dragDepth = 0; el.dropzone.classList.remove('dragging'); }
});
window.addEventListener('drop', (e) => {
  if (![...e.dataTransfer.types].includes('Files')) return;
  e.preventDefault();
  dragDepth = 0;
  el.dropzone.classList.remove('dragging');
  const file = [...e.dataTransfer.files].find((f) => f.type.startsWith('image/'));
  if (file) loadFile(file);
  else showError('No image file in that drop.');
});

window.addEventListener('paste', (e) => {
  const items = [...(e.clipboardData?.items || [])];
  const imageItem = items.find((i) => i.kind === 'file' && i.type.startsWith('image/'));
  if (imageItem) {
    e.preventDefault();
    const file = imageItem.getAsFile();
    if (file) loadFile(file);
    return;
  }
  const text = e.clipboardData?.getData('text/plain')?.trim();
  if (text && /^(https?:|data:image\/)/.test(text)) {
    e.preventDefault();
    loadFromUrl(text);
  }
});

el.modelSelect.addEventListener('change', () => { describeModel(); saveSettings(); });
el.scaleSelect.addEventListener('change', saveSettings);
el.backendSelect.addEventListener('change', () => { updateEstimate(); saveSettings(); });
el.tileSize.addEventListener('input', () => { el.tileValue.textContent = `${el.tileSize.value} px`; });
el.tileSize.addEventListener('change', saveSettings);
el.overlap.addEventListener('input', () => { el.overlapValue.textContent = `${el.overlap.value} px`; });
el.overlap.addEventListener('change', saveSettings);

el.run.addEventListener('click', startRun);
el.cancel.addEventListener('click', () => {
  state.worker?.postMessage({ type: 'cancel' });
  setStatus('Stopping after the current tile…');
});
el.clearCache.addEventListener('click', () => getWorker().postMessage({ type: 'evict' }));

el.tabCompare.addEventListener('click', () => setCompareMode('compare'));
el.tabResult.addEventListener('click', () => setCompareMode('result'));
el.downloadPng.addEventListener('click', () => download('image/png', 'png'));
el.downloadWebp.addEventListener('click', () => download('image/webp', 'webp', 0.95));

let dragging = false;
const splitFromEvent = (e) => {
  const rect = el.compare.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  setSplit(x / rect.width);
};
el.compare.addEventListener('pointerdown', (e) => {
  if (el.compare.classList.contains('result-only')) return;
  dragging = true;
  el.compare.setPointerCapture(e.pointerId);
  splitFromEvent(e);
});
el.compare.addEventListener('pointermove', (e) => { if (dragging) splitFromEvent(e); });
el.compare.addEventListener('pointerup', (e) => {
  dragging = false;
  try { el.compare.releasePointerCapture(e.pointerId); } catch { /* already released */ }
});
el.handle.addEventListener('keydown', (e) => {
  const current = Number(el.handle.getAttribute('aria-valuenow'));
  if (e.key === 'ArrowLeft') { e.preventDefault(); setSplit((current - 2) / 100); }
  if (e.key === 'ArrowRight') { e.preventDefault(); setSplit((current + 2) / 100); }
});

/* --------------------------------------------------------------- boot */

if (/Mac|iPhone|iPad/.test(navigator.platform || '')) el.pasteKey.textContent = '⌘';
buildModelSelect();
el.scaleSelect.value = state.settings.scale;
el.backendSelect.value = state.settings.backend;
el.tileSize.value = String(state.settings.tile);
el.overlap.value = String(state.settings.overlap);
el.tileValue.textContent = `${el.tileSize.value} px`;
el.overlapValue.textContent = `${el.overlap.value} px`;
describeModel();
getWorker().postMessage({ type: 'probe' });
