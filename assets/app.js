import {
  TOOLS, toolById, modelsForTask, defaultModelId, modelById, pickVariant, formatBytes,
} from './models.js';
import { encodeGreyPng } from './png.js';

const $ = (id) => document.getElementById(id);
const MAX_LONG_SIDE = 4096;        // 4× of this is 16384 px — the browser canvas ceiling
const MAX_RESULT_SIDE = 16384;     // a chained input (an upscaled result) may be this large
const STORE_KEY = 'drikus-upscale/settings';
const CALIB_KEY = 'drikus-upscale/calibration';
const REFINE_DEBOUNCE_MS = 150;

const el = {
  dropzone: $('dropzone'), fileInput: $('file-input'), browse: $('browse'),
  workspace: $('workspace'), thumb: $('thumb'), srcName: $('src-name'), srcDims: $('src-dims'),
  changeImage: $('change-image'), toolSwitch: $('tool-switch'),
  modelSelect: $('model-select'), modelBlurb: $('model-blurb'), modelMeta: $('model-meta'),
  modelEstimate: $('model-estimate'), scaleSelect: $('scale-select'), backendSelect: $('backend-select'),
  tileSize: $('tile-size'), tileValue: $('tile-value'), overlap: $('overlap'), overlapValue: $('overlap-value'),
  tileBlurb: $('tile-blurb'),
  radius: $('refine-radius'), radiusValue: $('radius-value'), threshold: $('refine-threshold'),
  thresholdValue: $('threshold-value'), decontaminate: $('refine-decontaminate'), bgColor: $('bg-color'),
  clearCache: $('clear-cache'), run: $('run'), cancel: $('cancel'), progress: $('progress'),
  barFill: $('bar-fill'), status: $('status'), error: $('error'), capsChip: $('caps-chip'),
  before: $('canvas-before'), after: $('canvas-after'), mask: $('canvas-mask'), clip: $('clip'), handle: $('handle'),
  compare: $('compare'), stage: $('stage'), tabCompare: $('tab-compare'), tabResult: $('tab-result'), tabMask: $('tab-mask'),
  downloadPng: $('download-png'), downloadWebp: $('download-webp'), downloadMask: $('download-mask'),
  useResult: $('use-result'), outDims: $('out-dims'), zoomToggle: $('zoom-toggle'),
  previewHint: $('preview-hint'), pasteKey: $('paste-key'), tagRight: $('tag-right'),
};

const HINTS = {
  upscale: 'The upscaled image is painted tile by tile as it is produced. Drag the divider to compare — and switch to <strong>100%</strong> above the preview, since "Fit" scales the added detail back down along with the image.',
  remove: 'The cutout appears in strips as edges are refined. Drag the divider to compare, or open <strong>Mask</strong> to inspect the alpha. Chain tools with <strong>Use as input</strong> — for example, remove the background and then upscale.',
};

const state = {
  worker: null,
  tool: 'upscale',
  source: null,          // { bitmap, width, height, name, imageData, hasAlpha, fromResult }
  alphaCtx: null,        // upscaled alpha, only when the source has transparency
  cutout: null,          // removal result at full res, transparent (the display canvas adds the background)
  result: null,          // { tool, done, name } — what the preview currently holds
  job: null,             // 'upscale' | 'remove' | 'refine' while the worker is busy
  running: false,
  startedAt: 0,
  modelScale: 4,
  generation: 0,         // latest cutout generation; older strips are ignored
  settings: loadSettings(),
  calibration: loadCalibration(),
  backendUsed: 'wasm',
  refineTimer: 0,
  refineSent: '',
};

function loadCalibration() {
  try { return JSON.parse(localStorage.getItem(CALIB_KEY) || '{}'); } catch { return {}; }
}

/* --------------------------------------------------------------- settings */

function loadSettings() {
  const defaults = {
    tool: 'upscale',
    models: { upscale: defaultModelId('upscale'), remove: defaultModelId('remove') },
    scale: '2', backend: 'auto', tile: 192, overlap: 16,
    refine: { radius: 8, threshold: 0, decontaminate: true },
    background: 'transparent', bgColor: '#ffffff',
  };
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    const models = { ...defaults.models, ...(saved.models || {}) };
    if (saved.model && !saved.models) models.upscale = saved.model;     // settings from before tools existed
    return { ...defaults, ...saved, models, refine: { ...defaults.refine, ...(saved.refine || {}) } };
  } catch {
    return defaults;
  }
}

function saveSettings() {
  state.settings = {
    tool: state.tool,
    models: { ...state.settings.models, [state.tool]: el.modelSelect.value },
    scale: el.scaleSelect.value,
    backend: el.backendSelect.value,
    tile: Number(el.tileSize.value),
    overlap: Number(el.overlap.value),
    refine: currentRefine(),
    background: backgroundMode(),
    bgColor: el.bgColor.value,
  };
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state.settings)); } catch { /* private mode */ }
}

/* --------------------------------------------------------------- worker */

function getWorker() {
  if (state.worker) return state.worker;
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => onWorkerMessage(e.data);
  worker.onerror = (e) => {
    showError(`The processing worker crashed: ${e.message || 'unknown error'}`);
    finishRun();
    state.worker?.terminate();
    state.worker = null;
  };
  state.worker = worker;
  return worker;
}

/* --------------------------------------------------------------- tools */

function buildToolSwitch() {
  el.toolSwitch.innerHTML = '';
  for (const tool of TOOLS) {
    const button = document.createElement('button');
    button.type = 'button';
    button.role = 'tab';
    button.dataset.tool = tool.id;
    button.textContent = tool.name;
    button.title = `${tool.name} (${tool.key.toUpperCase()})`;
    button.addEventListener('click', () => setTool(tool.id));
    el.toolSwitch.append(button);
  }
}

/** Switch tools. The loaded image and the current preview are kept. */
function setTool(id) {
  if (state.running) return;
  const tool = toolById(id);
  state.tool = tool.id;
  for (const button of el.toolSwitch.children) {
    button.setAttribute('aria-selected', String(button.dataset.tool === tool.id));
  }
  for (const node of document.querySelectorAll('[data-tool]')) {
    if (node.parentElement === el.toolSwitch) continue;
    node.hidden = node.dataset.tool !== tool.id;
  }
  el.run.textContent = tool.action;
  el.previewHint.innerHTML = HINTS[tool.id] || '';
  buildModelSelect();
  describeModel();
  updateResultControls();
  saveSettings();
}

function buildModelSelect() {
  el.modelSelect.innerHTML = '';
  for (const model of modelsForTask(state.tool)) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.recommended ? `${model.name}  ★` : model.name;
    el.modelSelect.append(option);
  }
  el.modelSelect.value = modelById(state.settings.models[state.tool], state.tool).id;
}

const selectedModel = () => modelById(el.modelSelect.value, state.tool);

/** Bytes this device will actually download for a model, given its backend. */
function downloadBytes(model) {
  return pickVariant(model, effectiveBackend(), state.caps).bytes;
}

function describeModel() {
  const model = selectedModel();
  el.modelMeta.innerHTML = '';
  if (model.recommended) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'Recommended';
    el.modelMeta.append(badge);
  }
  const shape = model.task === 'remove' ? `${model.inputSize}² segmentation` : `${model.scale}×`;
  el.modelMeta.append(document.createTextNode(`${shape} · ${formatBytes(downloadBytes(model))}`));
  el.modelBlurb.textContent = model.blurb;
  if (model.task === 'upscale') updateTileControls(model);
  updateEstimate();
}

/**
 * Most models tolerate any tile size the user picks. Swin2SR was exported at
 * one exact input resolution -- its attention mask is baked in for that shape,
 * so the sliders are locked to it here rather than left free to produce a
 * silently wrong result at any other tile size (see worker.js).
 */
function updateTileControls(model) {
  const locked = Boolean(model.fixedTile);
  el.tileSize.disabled = locked;
  el.overlap.disabled = locked;
  if (locked) {
    el.tileValue.textContent = `${model.fixedTile} px (fixed)`;
    el.overlapValue.textContent = `${model.fixedContextPad} px (fixed)`;
    el.tileBlurb.textContent = 'This model was exported at one fixed tile size, so it ignores the sliders above.';
  } else {
    el.tileValue.textContent = `${el.tileSize.value} px`;
    el.overlapValue.textContent = `${el.overlap.value} px`;
    el.tileBlurb.textContent = 'Tiles keep memory bounded. Smaller tiles are safer on phones; the overlap is cropped away after inference so seams stay invisible.';
  }
}

/** Which backend a run would actually use, given the picker and the hardware. */
function effectiveBackend() {
  const choice = el.backendSelect.value;
  if (choice === 'wasm') return 'wasm';
  if (choice === 'webgpu') return 'webgpu';
  return state.caps?.webgpu ? 'webgpu' : 'wasm';
}

/**
 * Upscaling throughput in "cost-weighted megapixels per second". The seeds
 * are rough measurements; every completed run replaces them with what this
 * machine actually did, so the estimate sharpens after the first upscale.
 */
function throughput(backend) {
  const calibrated = state.calibration[backend];
  if (calibrated) return calibrated;
  if (backend === 'webgpu') return 0.4;
  return 0.007 * Math.max(1, state.caps?.threads || 1);
}

/** Seconds for one segmentation pass: measured if we have it, else a seed scaled for threads. */
function segmentationSeconds(model, backend) {
  const measured = state.calibration[`remove:${model.id}:${backend}`];
  if (measured) return measured;
  const seed = model.seconds[backend];
  return backend === 'wasm' ? seed * (4 / Math.max(1, state.caps?.threads || 1)) : seed;
}

const prettySeconds = (s) => (s < 2 ? 'a second or two'
  : s < 90 ? `about ${Math.round(s)} s` : `about ${Math.round(s / 60)} min`);

function updateEstimate() {
  if (!state.source) { el.modelEstimate.textContent = ''; return; }
  const model = selectedModel();
  const backend = effectiveBackend();
  const mpx = (state.source.width * state.source.height) / 1e6;
  const download = `${formatBytes(downloadBytes(model))} download on first use`;
  let seconds;
  let text;
  if (model.task === 'remove') {
    // Inference is fixed-size; refinement scales with the image.
    seconds = segmentationSeconds(model, backend) + mpx * 0.25;
    const measured = state.calibration[`remove:${model.id}:${backend}`] ? '' : ' (rough guess until the first run)';
    const singleThread = backend === 'wasm' && state.caps && state.caps.threads === 1
      ? ' WASM is single-threaded here because the page is not cross-origin isolated, so this will be slow.' : '';
    text = `${model.inputSize}² segmentation · ${download} · ${prettySeconds(seconds)} on ${backend.toUpperCase()}${measured}.${singleThread}`;
  } else {
    seconds = (mpx * model.cost) / throughput(backend);
    const measured = state.calibration[backend] ? '' : ' (rough guess until the first run)';
    text = `${mpx.toFixed(2)} Mpx input · ${download} · ${prettySeconds(seconds)} on ${backend.toUpperCase()}${measured}.`;
    if (Math.max(state.source.width, state.source.height) > MAX_LONG_SIDE) {
      text = `This image is larger than ${MAX_LONG_SIDE} px on its long side, so a 4× result would not fit in a browser canvas. Remove its background, or start from a smaller image.`;
      seconds = Infinity;
    }
  }
  el.modelEstimate.textContent = text;
  el.modelEstimate.classList.toggle('slow', seconds > 60);
}

function recordCalibration(backend, seconds) {
  const model = modelById(el.modelSelect.value, 'upscale');
  const mpx = (state.source.width * state.source.height) / 1e6;
  if (seconds < 0.4) return;                       // too short to measure anything
  const observed = (mpx * model.cost) / seconds;
  const previous = state.calibration[backend];
  state.calibration[backend] = previous ? previous * 0.4 + observed * 0.6 : observed;
  persistCalibration();
}

function recordSegmentation(modelId, backend, seconds) {
  const key = `remove:${modelId}:${backend}`;
  const previous = state.calibration[key];
  state.calibration[key] = previous ? previous * 0.4 + seconds * 0.6 : seconds;
  persistCalibration();
}

function persistCalibration() {
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(state.calibration)); } catch { /* ignore */ }
}

function updateCapsChip(caps) {
  state.caps = caps;
  if (caps.webgpu) {
    el.capsChip.textContent = caps.shaderF16 ? 'WebGPU · f16' : 'WebGPU';
    el.capsChip.className = 'chip chip-ok';
    el.capsChip.title = `${caps.adapter ? `WebGPU via ${caps.adapter}` : 'WebGPU available'}. `
      + (caps.shaderF16
        ? 'The adapter supports shader-f16, so background removal downloads half-size fp16 weights.'
        : 'No shader-f16 support, so background removal uses the full-size fp32 weights.');
  } else {
    el.capsChip.textContent = `WASM · ${caps.threads} thread${caps.threads === 1 ? '' : 's'}`;
    el.capsChip.className = 'chip chip-muted';
    el.capsChip.title = caps.crossOriginIsolated
      ? 'No WebGPU adapter — running on multi-threaded WebAssembly'
      : 'No WebGPU adapter, and the page is not cross-origin isolated, so WASM runs single-threaded';
  }
  describeModel();
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
  loadBitmap(bitmap, file.name || 'pasted-image.png', { fromResult: false });
}

/**
 * Make a decoded image the current source. A chained result (`fromResult`)
 * may be up to 16384 px — the most an upscale can produce — while a fresh
 * input is held to 4096 px so its 4× upscale still fits a canvas.
 */
function loadBitmap(bitmap, name, { fromResult }) {
  if (state.running) return;
  const limit = fromResult ? MAX_RESULT_SIDE : MAX_LONG_SIDE;
  if (Math.max(bitmap.width, bitmap.height) > limit) {
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

  state.source?.bitmap?.close?.();
  state.source = { bitmap, width: bitmap.width, height: bitmap.height, name, imageData, hasAlpha, fromResult };
  state.alphaCtx = null;
  state.cutout = null;
  state.result = null;
  state.worker?.postMessage({ type: 'release-image' });

  el.srcName.textContent = name;
  el.srcDims.textContent = `${bitmap.width} × ${bitmap.height}${hasAlpha ? ' · has transparency' : ''}${fromResult ? ' · from the previous result' : ''}`;
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

/** Feed the current result back in as the source, to chain tools (FR-13). */
async function useResultAsInput() {
  if (!state.result?.done || state.running) return;
  const bitmap = await createImageBitmap(el.after);
  const base = state.source.name.replace(/\.[^.]+$/, '') || 'image';
  loadBitmap(bitmap, `${base}${resultSuffix()}.png`, { fromResult: true });
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
  el.compare.classList.remove('cutout');
  updateResultControls();
  setCompareMode('compare');
  setSplit(0.5);
  el.compare.classList.remove('native');
  el.zoomToggle.textContent = 'Fit ▾';
}

function sizeCanvases(width, height) {
  for (const canvas of [el.before, el.after]) {
    canvas.width = width;
    canvas.height = height;
  }
}

/** The result canvas sits on top and is revealed to the right of the handle. */
function setSplit(fraction) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  el.clip.style.clipPath = `inset(0 0 0 ${pct}%)`;
  el.handle.style.left = `${pct}%`;
  el.handle.setAttribute('aria-valuenow', String(Math.round(pct)));
}

function setCompareMode(mode) {
  if (mode === 'mask' && !(state.result?.tool === 'remove' && state.cutout)) mode = 'compare';
  el.compare.classList.toggle('result-only', mode === 'result');
  el.compare.classList.toggle('mask-only', mode === 'mask');
  el.tabResult.classList.toggle('active', mode === 'result');
  el.tabCompare.classList.toggle('active', mode === 'compare');
  el.tabMask.classList.toggle('active', mode === 'mask');
  state.compareMode = mode;
  if (mode === 'mask') drawMaskView();
}

/** Enable result-dependent buttons and label the comparison for what's shown. */
function updateResultControls() {
  const ready = Boolean(state.result?.done) && !state.running;
  el.downloadPng.disabled = !ready;
  el.downloadWebp.disabled = !ready;
  el.useResult.disabled = !ready;
  const cutout = state.result?.tool === 'remove';
  el.downloadMask.disabled = !(ready && cutout);
  el.tabMask.disabled = !(cutout && state.cutout);
  el.tagRight.textContent = state.result ? (cutout ? 'cutout' : 'upscaled')
    : state.tool === 'remove' ? 'cutout' : 'upscaled';
}

const resultSuffix = () => (state.result?.tool === 'remove'
  ? '_nobg'
  : `_upscaled_${state.result?.scale || el.scaleSelect.value}x`);

/* --------------------------------------------------------------- run */

function startRun() {
  if (!state.source || state.running) return;
  hideError();
  saveSettings();
  if (state.tool === 'remove') startRemove();
  else startUpscale();
}

function beginJob(job, label) {
  state.job = job;
  state.running = true;
  state.startedAt = performance.now();
  el.run.disabled = true;
  for (const button of el.toolSwitch.children) button.disabled = true;
  el.cancel.hidden = false;
  el.progress.hidden = false;
  setBar(0);
  setStatus(label);
  updateResultControls();
}

function startUpscale() {
  const model = selectedModel();
  const { width, height } = state.source;
  if (Math.max(width, height) > MAX_LONG_SIDE) {
    showError(`Upscaling needs the long side under ${MAX_LONG_SIDE} px so the 4× result fits inside a browser canvas; this image is ${width}×${height}.`);
    return;
  }
  state.modelScale = model.scale;
  state.result = { tool: 'upscale', done: false, scale: el.scaleSelect.value };
  state.cutout = null;
  el.compare.classList.remove('cutout');

  sizeCanvases(width * model.scale, height * model.scale);
  const bctx = el.before.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'high';
  bctx.drawImage(state.source.bitmap, 0, 0, el.before.width, el.before.height);
  el.after.getContext('2d').clearRect(0, 0, el.after.width, el.after.height);

  if (state.source.hasAlpha) prepareAlpha(width * model.scale, height * model.scale);
  beginJob('upscale', `Loading ${model.name}…`);

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

function startRemove() {
  const model = selectedModel();
  const { width, height } = state.source;
  state.result = { tool: 'remove', done: false, modelId: model.id };
  if (state.compareMode === 'mask') setCompareMode('compare');

  sizeCanvases(width, height);
  const bctx = el.before.getContext('2d');
  bctx.drawImage(state.source.bitmap, 0, 0);
  el.after.getContext('2d').clearRect(0, 0, width, height);
  state.cutout = document.createElement('canvas');
  state.cutout.width = width;
  state.cutout.height = height;
  el.compare.classList.add('cutout');

  beginJob('remove', `Loading ${model.name}…`);
  const refine = currentRefine();
  state.refineSent = JSON.stringify(refine);
  const pixels = state.source.imageData.data.slice().buffer;
  getWorker().postMessage({
    type: 'remove', pixels, width, height, modelId: model.id, backend: el.backendSelect.value, refine,
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

const STAGE_LABELS = {
  hash: 'Reading the image…',
  compile: 'Compiling the model…',
  infer: 'Segmenting…',
  refine: 'Refining edges',
  decontaminate: 'Refining edges and cleaning colours',
};

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
      setBar(pct);
      setStatus(msg.cached ? 'Model loaded from cache.' : `Downloading model… ${pct}% (${formatBytes(msg.loaded)} of ${formatBytes(msg.total)})`);
      break;
    }
    case 'stage':
      onStage(msg);
      break;
    case 'ready':
      // Also fires mid-run if the worker has to switch backends.
      state.backendUsed = msg.backend;
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
    case 'mask':
      onMask(msg);
      break;
    case 'cutout-start':
      state.generation = msg.generation;
      break;
    case 'cutout':
      paintCutoutStrip(msg);
      break;
    case 'done':
      // Exposed for tools/smoke-test.mjs: proves cache hits and refines never touch the model.
      window.drikusStats = { inferences: msg.inferences };
      if (state.job === 'upscale') completeUpscale();
      else if (msg.generation === state.generation) completeRemove(msg);
      break;
    case 'cancelled':
      setStatus('Cancelled.');
      finishRun();
      break;
    case 'evicted':
      setStatus('Cached models and masks cleared.');
      break;
    case 'error':
      showError(msg.message);
      finishRun();
      break;
  }
}

function onStage(msg) {
  const label = STAGE_LABELS[msg.name] || msg.name;
  el.barFill.classList.toggle('indeterminate', msg.name === 'infer' || msg.name === 'compile');
  if (msg.name === 'refine' || msg.name === 'decontaminate') {
    setBar(Math.round((msg.done / msg.total) * 100));
    setStatus(`${label} — strip ${msg.done + 1} of ${msg.total}…`);
  } else {
    if (msg.name === 'infer') state.inferStartedAt = performance.now();
    setStatus(msg.name === 'infer' ? `Segmenting on ${state.backendUsed.toUpperCase()}…` : label);
  }
}

/**
 * A raw S×S mask: paint it straight away as a cheap preview (scaled by the
 * GPU, multiplied onto the source), so something useful shows while edges
 * are refined — or, for a fast-model preview, while the main model runs.
 */
function onMask(msg) {
  if (!state.cutout || state.job !== 'remove') return;
  const size = msg.size;
  const mask = new Uint8Array(msg.mask);
  const small = document.createElement('canvas');
  small.width = size;
  small.height = size;
  const image = new ImageData(size, size);
  for (let i = 0; i < mask.length; i++) image.data[i * 4 + 3] = mask[i];
  small.getContext('2d').putImageData(image, 0, 0);

  const { width, height } = state.source;
  const ctx = state.cutout.getContext('2d');
  ctx.globalCompositeOperation = 'copy';
  ctx.drawImage(state.source.bitmap, 0, 0, width, height);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, width, height);
  ctx.globalCompositeOperation = 'source-over';
  repaintResult(0, height);

  if (msg.preview) {
    setStatus('Showing a quick U²-Netp preview while the full model runs…');
    return;
  }
  if (!msg.cached && msg.seconds) recordSegmentation(state.result.modelId, msg.backend, msg.seconds);
  state.result.cached = msg.cached;
  state.result.backend = msg.backend;
  state.result.maskReady = true;      // the worker now holds it for refine re-runs
  el.barFill.classList.remove('indeterminate');
}

function paintCutoutStrip(msg) {
  if (msg.generation !== state.generation || !state.cutout) return;
  const rgba = new Uint8ClampedArray(msg.pixels);
  state.cutout.getContext('2d').putImageData(new ImageData(rgba, msg.width, msg.h), 0, msg.y);
  repaintResult(msg.y, msg.h);
  setBar(Math.round((msg.done / msg.total) * 100));
}

const backgroundMode = () => document.querySelector('input[name="bg"]:checked')?.value || 'transparent';

/** Copy rows of the transparent cutout to the visible canvas, over the chosen background. */
function repaintResult(y, h) {
  if (!state.cutout) return;
  const ctx = el.after.getContext('2d');
  const { width } = state.cutout;
  if (backgroundMode() === 'colour') {
    ctx.fillStyle = el.bgColor.value;
    ctx.fillRect(0, y, width, h);
  } else {
    ctx.clearRect(0, y, width, h);
  }
  ctx.drawImage(state.cutout, 0, y, width, h, 0, y, width, h);
}

function maskBytes() {
  const { width, height } = state.cutout;
  const data = state.cutout.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, width, height).data;
  const grey = new Uint8Array(width * height);
  for (let i = 0; i < grey.length; i++) grey[i] = data[i * 4 + 3];
  return grey;
}

function drawMaskView() {
  if (!state.cutout) return;
  const { width, height } = state.cutout;
  const grey = maskBytes();
  const image = new ImageData(width, height);
  for (let i = 0; i < grey.length; i++) {
    const d = i * 4;
    image.data[d] = image.data[d + 1] = image.data[d + 2] = grey[i];
    image.data[d + 3] = 255;
  }
  el.mask.width = width;
  el.mask.height = height;
  el.mask.getContext('2d').putImageData(image, 0, 0);
}

function completeRemove(msg) {
  const elapsed = (performance.now() - state.startedAt) / 1000;
  const refineOnly = state.job === 'refine';
  state.result.done = true;
  el.outDims.textContent = `${el.after.width} × ${el.after.height}`;
  if (state.compareMode === 'mask') drawMaskView();
  const how = state.result.cached ? ' (mask from cache)' : ` on ${(state.result.backend || state.backendUsed).toUpperCase()}`;
  setStatus(refineOnly
    ? `Edges refined in ${msg.refineSeconds.toFixed(1)} s.`
    : `Done in ${elapsed.toFixed(1)} s${how}.`);
  finishRun();
  // Slider moves made while the model ran are applied now.
  if (JSON.stringify(currentRefine()) !== state.refineSent) scheduleRefine(0);
}

function paintTile(msg) {
  const rgba = new Uint8ClampedArray(msg.pixels);
  if (state.alphaCtx) {
    const alpha = state.alphaCtx.getImageData(msg.x, msg.y, msg.w, msg.h).data;
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = alpha[i];
  }
  el.after.getContext('2d').putImageData(new ImageData(rgba, msg.w, msg.h), msg.x, msg.y);

  const pct = Math.round((msg.done / msg.total) * 100);
  setBar(pct);
  const elapsed = (performance.now() - state.startedAt) / 1000;
  const remaining = (elapsed / msg.done) * (msg.total - msg.done);
  setStatus(`Tile ${msg.done} of ${msg.total} · ${elapsed.toFixed(1)} s elapsed · ~${remaining.toFixed(0)} s left`);
}

function completeUpscale() {
  const target = Number(el.scaleSelect.value);
  if (target !== state.modelScale) resampleTo(target);
  state.result.done = true;
  el.outDims.textContent = `${el.after.width} × ${el.after.height}`;
  const elapsed = (performance.now() - state.startedAt) / 1000;
  recordCalibration(state.backendUsed, elapsed);
  setStatus(`Done in ${elapsed.toFixed(1)} s on ${state.backendUsed.toUpperCase()}.`);
  finishRun();
  updateEstimate();
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
  state.job = null;
  el.run.disabled = false;
  for (const button of el.toolSwitch.children) button.disabled = false;
  el.cancel.hidden = true;
  el.barFill.classList.remove('indeterminate');
  updateResultControls();
}

/* --------------------------------------------------------------- refinement */

function currentRefine() {
  return {
    radius: Number(el.radius.value),
    threshold: Number(el.threshold.value),
    decontaminate: el.decontaminate.checked,
  };
}

function updateRefineLabels() {
  el.radiusValue.textContent = `${el.radius.value} px`;
  el.thresholdValue.textContent = Number(el.threshold.value) === 0 ? 'off' : Number(el.threshold.value).toFixed(2);
}

/** Re-run only refinement from the worker's held mask (UI-4): no model work. */
function scheduleRefine(delay = REFINE_DEBOUNCE_MS) {
  clearTimeout(state.refineTimer);
  state.refineTimer = setTimeout(() => {
    if (state.running || state.result?.tool !== 'remove' || !state.result.maskReady) return;
    const refine = currentRefine();
    state.refineSent = JSON.stringify(refine);
    state.result.done = false;
    beginJob('refine', 'Refining edges…');
    getWorker().postMessage({ type: 'refine', refine });
  }, delay);
}

function onRefineInput() {
  updateRefineLabels();
  saveSettings();
  scheduleRefine();
}

function onBackgroundChange() {
  saveSettings();
  if (state.result?.tool === 'remove' && state.cutout) repaintResult(0, state.cutout.height);
}

/* --------------------------------------------------------------- helpers */

function setBar(pct) {
  el.barFill.style.width = `${pct}%`;
}

function setStatus(text, kind) {
  el.status.textContent = text;
  el.status.className = `status${kind === 'warn' ? ' warn' : ''}`;
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = false;
}

function hideError() { el.error.hidden = true; }

function saveBlob(blob, filename) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 10000);
}

const baseName = () => state.source.name.replace(/\.[^.]+$/, '') || 'image';

function download(type, extension, quality) {
  el.after.toBlob((blob) => {
    if (!blob) { showError('The browser refused to encode the result.'); return; }
    saveBlob(blob, `${baseName()}${resultSuffix()}.${extension}`);
  }, type, quality);
}

async function downloadMask() {
  if (!state.cutout) return;
  const blob = await encodeGreyPng(maskBytes(), state.cutout.width, state.cutout.height);
  saveBlob(blob, `${baseName()}_mask.png`);
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

// Tool shortcuts (U, R, …) whenever focus isn't in a text field or control.
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const target = e.target;
  if (target instanceof HTMLElement && (target.isContentEditable
    || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName))) return;
  const tool = TOOLS.find((t) => t.key === e.key.toLowerCase());
  if (tool && tool.id !== state.tool) { e.preventDefault(); setTool(tool.id); }
});

el.modelSelect.addEventListener('change', () => { describeModel(); saveSettings(); });
el.scaleSelect.addEventListener('change', saveSettings);
el.backendSelect.addEventListener('change', () => { describeModel(); saveSettings(); });
el.tileSize.addEventListener('input', () => { el.tileValue.textContent = `${el.tileSize.value} px`; });
el.tileSize.addEventListener('change', saveSettings);
el.overlap.addEventListener('input', () => { el.overlapValue.textContent = `${el.overlap.value} px`; });
el.overlap.addEventListener('change', saveSettings);
el.radius.addEventListener('input', onRefineInput);
el.threshold.addEventListener('input', onRefineInput);
el.decontaminate.addEventListener('change', onRefineInput);
for (const radio of document.querySelectorAll('input[name="bg"]')) radio.addEventListener('change', onBackgroundChange);
el.bgColor.addEventListener('input', () => {
  const colour = document.querySelector('input[name="bg"][value="colour"]');
  if (!colour.checked) colour.checked = true;
  onBackgroundChange();
});

el.run.addEventListener('click', startRun);
el.cancel.addEventListener('click', () => {
  state.worker?.postMessage({ type: 'cancel' });
  setStatus(state.job === 'upscale' ? 'Stopping after the current tile…' : 'Stopping…');
});
el.clearCache.addEventListener('click', () => getWorker().postMessage({ type: 'evict' }));

el.tabCompare.addEventListener('click', () => setCompareMode('compare'));
el.tabResult.addEventListener('click', () => setCompareMode('result'));
el.tabMask.addEventListener('click', () => setCompareMode('mask'));
el.zoomToggle.addEventListener('click', () => {
  const native = el.compare.classList.toggle('native');
  el.zoomToggle.textContent = native ? '100% ▾' : 'Fit ▾';
});
el.downloadPng.addEventListener('click', () => download('image/png', 'png'));
el.downloadWebp.addEventListener('click', () => download('image/webp', 'webp', 0.95));
el.downloadMask.addEventListener('click', downloadMask);
el.useResult.addEventListener('click', useResultAsInput);

let dragging = false;
const splitFromEvent = (e) => {
  const rect = el.compare.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  setSplit(x / rect.width);
};
el.compare.addEventListener('pointerdown', (e) => {
  if (el.compare.classList.contains('result-only') || el.compare.classList.contains('mask-only')) return;
  e.preventDefault();      // otherwise a fast drag starts a text/image selection instead of moving the handle
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
el.scaleSelect.value = state.settings.scale;
el.backendSelect.value = state.settings.backend;
el.tileSize.value = String(state.settings.tile);
el.overlap.value = String(state.settings.overlap);
el.tileValue.textContent = `${el.tileSize.value} px`;
el.overlapValue.textContent = `${el.overlap.value} px`;
el.radius.value = String(state.settings.refine.radius);
el.threshold.value = String(state.settings.refine.threshold);
el.decontaminate.checked = state.settings.refine.decontaminate;
el.bgColor.value = state.settings.bgColor;
document.querySelector(`input[name="bg"][value="${state.settings.background === 'colour' ? 'colour' : 'transparent'}"]`).checked = true;
updateRefineLabels();
buildToolSwitch();
setTool(state.settings.tool);
getWorker().postMessage({ type: 'probe' });
