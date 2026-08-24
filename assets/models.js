// Model catalogue. `cost` is a rough per-pixel compute weight relative to the
// compact models, used to pick sane defaults and to warn before slow jobs.
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
];

export const DEFAULT_MODEL_ID = MODELS.find((m) => m.recommended).id;

export const modelById = (id) => MODELS.find((m) => m.id === id) || MODELS.find((m) => m.recommended);

export const formatBytes = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} kB`);
