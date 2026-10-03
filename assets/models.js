// Tools the app offers. Each one owns a set of models (matched on `task`), its
// own controls in index.html (`data-tool` sections) and a keyboard shortcut.
export const TOOLS = [
  { id: 'upscale', name: 'Upscale', key: 'u', action: 'Upscale' },
  { id: 'remove', name: 'Remove background', key: 'r', action: 'Remove background' },
];

export const toolById = (id) => TOOLS.find((t) => t.id === id) || TOOLS[0];

// Model catalogue. For upscalers `cost` is a rough per-pixel compute weight
// relative to the compact models, used to pick sane defaults and to warn
// before slow jobs. Segmentation models run once per image at a fixed
// `inputSize`, so they carry seed timings per backend instead (`seconds`),
// replaced by measurements after the first run.
//
// A model either has one `file` or several `variants`; the worker picks the
// variant from the backend and adapter features (see pickVariant), never the
// user. A variant's `files` are the graph followed by its external-data
// shards, and `sha256` is the graph's hash, which keys the mask cache so a
// re-exported model never serves stale masks.
export const MODELS = [
  {
    id: 'realesr-general-x4v3-balanced',
    name: 'General ×4 — balanced',
    file: 'models/realesr-general-x4v3-balanced.onnx',
    bytes: 4866422,
    scale: 4,
    cost: 1,
    recommended: true,
    blurb: 'Real-ESRGAN general v3, half-blended with its denoise variant. The best all-rounder for photos, screenshots and JPEG-mangled images.',
  },
  {
    id: 'realesr-general-x4v3-detail',
    name: 'General ×4 — max detail',
    file: 'models/realesr-general-x4v3-detail.onnx',
    bytes: 4866422,
    scale: 4,
    cost: 1,
    blurb: 'Same network, no denoise blending. Keeps the most texture, but also keeps grain and compression noise.',
  },
  {
    id: 'realesr-general-x4v3-denoise',
    name: 'General ×4 — max denoise',
    file: 'models/realesr-general-x4v3-denoise.onnx',
    bytes: 4866422,
    scale: 4,
    cost: 1,
    blurb: 'The full WDN variant. Reach for this on heavily compressed or noisy sources; it smooths fine texture along with the artefacts.',
  },
  {
    id: 'realesr-animevideov3-x4',
    name: 'Anime video ×4 — fast',
    file: 'models/realesr-animevideov3-x4.onnx',
    bytes: 2492902,
    scale: 4,
    cost: 0.5,
    blurb: 'Smallest and quickest model here. Tuned for anime and flat-shaded art; washes out photographic texture.',
  },
  {
    id: 'realesrgan-x4plus-anime-6b',
    name: 'Anime / art ×4 — 6B',
    file: 'models/realesrgan-x4plus-anime-6b.onnx',
    bytes: 17939969,
    scale: 4,
    cost: 8,
    blurb: 'RRDBNet with 6 blocks. Crisp line art and clean gradients on illustrations, manga scans and game sprites.',
  },
  {
    id: 'realesrgan-x4plus',
    name: 'Photo ×4 — x4plus (heavy)',
    file: 'models/realesrgan-x4plus.onnx',
    bytes: 67051644,
    scale: 4,
    cost: 30,
    blurb: 'The full 23-block Real-ESRGAN. Strongest artefact removal and detail synthesis on real photos — and roughly 30× the compute of the general model.',
  },
  {
    id: 'swin2sr-compressed-x4',
    name: 'Photo ×4 — Swin2SR compressed (slowest)',
    file: 'models/swin2sr-compressed-x4.onnx',
    bytes: 57149816,
    scale: 4,
    cost: 28,
    // Window attention over the whole tile still benefits from some overlap
    // to hide seams between tiles inferred independently of one another, but
    // needs far less than the CNNs above (see worker.js). The exported graph
    // is traced at this exact size (input tile + 2*contextPad on each side);
    // any other shape would silently produce an incorrect attention mask, so
    // the worker forces this tile geometry and the UI locks the sliders.
    fixedTile: 96,
    fixedContextPad: 16,
    blurb: 'A Swin transformer trained specifically on compressed/JPEG-degraded inputs (Conde & Choi et al., 2022) rather than a GAN. Different failure modes than the Real-ESRGAN models above — worth trying when they still look artefact-y.',
  },

  /* ------------------------------------------------ background removal */
  {
    id: 'birefnet-lite',
    task: 'remove',
    name: 'BiRefNet lite — best edges',
    inputSize: 1024,
    seconds: { webgpu: 1.0, wasm: 12 },        // wasm seeds are for 4 threads
    recommended: true,
    // Shown immediately when its model is already cached, while this one runs.
    preview: 'u2netp',
    variants: [
      { precision: 'fp16', requires: 'shader-f16', backends: ['webgpu'],
        files: ['models/birefnet-lite.fp16.onnx'], bytes: 92378509,
        sha256: 'c81c39c4f6797c191010389aa34345bd1adfdf529c4c1dee2b10705582ea6542' },
      { precision: 'fp32', backends: ['webgpu', 'wasm'],
        files: ['models/birefnet-lite.fp32.onnx', 'models/birefnet-lite.fp32.data0', 'models/birefnet-lite.fp32.data1'],
        bytes: 181900419, sha256: 'd11b7f7a596cee890cc5bfebbba1cecffc5961b1b792dd9b716d471e05f40826' },
    ],
    blurb: 'Bilateral-reference segmentation (Zheng et al., 2024) on a Swin-tiny backbone at 1024 px. Clean edges on hair, fur and fine structures; a sizeable one-time download.',
  },
  {
    id: 'u2netp',
    task: 'remove',
    name: 'U²-Netp — fast',
    inputSize: 320,
    seconds: { webgpu: 0.15, wasm: 0.5 },
    variants: [
      { precision: 'fp32', backends: ['webgpu', 'wasm'], files: ['models/u2netp.onnx'], bytes: 4632273,
        sha256: '9bb0a68759316ce3ebcf55c2cd70f63fbaca54d1e81ecc02ef1062a4787ab25a' },
    ],
    blurb: 'A 4.6 MB salient-object network (Qin et al., 2020) at 320 px. Near-instant on any device; softer, less precise edges.',
  },
];

for (const model of MODELS) model.task = model.task || 'upscale';

export const modelsForTask = (task) => MODELS.filter((m) => m.task === task);

export const defaultModelId = (task) => (modelsForTask(task).find((m) => m.recommended) || modelsForTask(task)[0]).id;

export const DEFAULT_MODEL_ID = defaultModelId('upscale');

/** Look a model up; unknown ids fall back to the recommended model of `task`. */
export const modelById = (id, task) => MODELS.find((m) => m.id === id && (!task || m.task === task))
  || MODELS.find((m) => m.id === defaultModelId(task || 'upscale'));

/** Every model as a list of variants, so single-file upscalers need no special case. */
export const variantsOf = (model) => model.variants
  || [{ precision: 'fp32', backends: ['webgpu', 'wasm'], files: [model.file], bytes: model.bytes, sha256: '' }];

/**
 * The variant a backend should load: fp16 only on WebGPU adapters with
 * shader-f16, otherwise the first variant the backend can run.
 */
export function pickVariant(model, backend, caps) {
  const usable = variantsOf(model).filter((v) => v.backends.includes(backend)
    && (!v.requires || (v.requires === 'shader-f16' && backend === 'webgpu' && caps?.shaderF16)));
  return usable[0] || variantsOf(model)[variantsOf(model).length - 1];
}

export const formatBytes = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} kB`);
