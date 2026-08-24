# drikus-upscale

AI image upscaling that runs entirely in the browser. Drop in a small or
JPEG-mangled picture, pick a model, and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
reconstructs it at 4× on your GPU (WebGPU) or CPU (WebAssembly SIMD + threads).
Nothing is uploaded — there is no server side.

**→ https://drikusroor.github.io/drikus-upscale/**

## What it does

- **Three ways in**: drag and drop anywhere on the page, paste with <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>V</kbd>,
  or the regular file picker. Pasting an image URL works too when the host allows cross-origin reads.
- **Six models** to choose from, with a recommended default (see below).
- **WebGPU when available, WASM otherwise**, decided automatically and overridable.
  If WebGPU fails to compile a model, the app falls back to WASM by itself.
- **Tiled inference in a worker**, so the UI stays responsive, memory stays bounded
  and the result paints in progressively, tile by tile.
- **Before/after slider**, PNG and WebP download, alpha channel preserved.
- **Models are cached** in the Cache API after the first download, so repeat runs
  are instant and the app works offline.

## Models

All six are exported from the official Real-ESRGAN checkpoints to ONNX with fully
dynamic input shapes. Every one of them outputs 4×; the "output size" selector
resamples that down when you want 3×, 2× or a same-size cleanup.

| Model | Size | Relative cost | Good for |
|---|---|---|---|
| **General ×4 — balanced** *(default)* | 4.9 MB | 1× | Photos, screenshots, JPEG artefacts — the all-rounder |
| General ×4 — max detail | 4.9 MB | 1× | Clean sources where you want every bit of texture |
| General ×4 — max denoise | 4.9 MB | 1× | Heavily compressed or noisy sources |
| Anime video ×4 — fast | 2.5 MB | 0.5× | Anime, flat-shaded art; the quickest option |
| Anime / art ×4 — 6B | 17.9 MB | 8× | Illustrations, manga scans, sprites |
| Photo ×4 — x4plus (heavy) | 67.1 MB | 30× | Best photo restoration, if you can wait |

The three "General" entries are the same `realesr-general-x4v3` network: the plain
weights, the WDN (denoise) weights, and — for the default — a 50/50 deep network
interpolation of the two, which is what `--denoise_strength 0.5` produces upstream.

### Rough speed

Time scales with input megapixels × the model's relative cost. On a 4-thread
WebAssembly build in CI, the general model does ~0.03 cost-weighted Mpx/s;
WebGPU is typically an order of magnitude faster. The app measures its own
throughput after each run and uses that for subsequent estimates, so the
"about N s" line gets accurate quickly.

## How it works

```
image → ImageBitmap → RGB Float32 NCHW → [tile + overlap] → ONNX Runtime Web → tile ×4 → canvas
```

- **Tiling.** The image is cut into square tiles (192 px by default) and each tile is
  inferred with an extra border of context (16 px by default) that is cropped away
  afterwards. Because the crop is inside the receptive field, seams are invisible —
  the smoke test measures the gradient across tile boundaries and finds it *lower*
  than the image's average gradient.
- **Uniform tensor shapes.** Edge tiles are not ragged: out-of-bounds reads clamp to
  the border pixel so every tile is inferred at exactly the same shape. That stops
  WebGPU from recompiling shaders for the right and bottom edges.
- **Distinct output dim params.** The exported graphs name the output axes
  `height_out`/`width_out`, *not* `height`/`width`. A repeated `dim_param` in ONNX is
  an assertion that the two axes are equal, so reusing the input's names claims a
  4×-larger output is the same size as its input. ONNX Runtime's allocation planner
  believes it, aliases the output onto the input buffer, and fails at run time with
  `Shape mismatch attempting to re-use buffer` — on WebGPU, where the planner is
  strict. The graphs also end in a `Clip` to [0, 1], which keeps the output a
  distinct tensor from the input as well as bounding the values.
- **Backend recovery.** If WebGPU still throws mid-run, the worker rebuilds the
  session once and then finishes the job on WASM rather than leaving a dead end.
  Sessions are pinned to one tile shape and rebuilt when it changes.
- **Alpha.** Real-ESRGAN is a 3-channel network. If the source has transparency, the
  alpha plane is upscaled separately with high-quality canvas resampling and
  recomposited onto the model's RGB output.
- **Cross-origin isolation.** GitHub Pages cannot send COOP/COEP headers, and WASM
  threads need them, so `coi-serviceworker.js` registers a service worker that adds
  them to same-origin responses and reloads once. If that fails the app still runs,
  single-threaded.
- **Canvas ceiling.** Browsers cap canvases at 16384 px, so inputs are limited to
  4096 px on the long side.

## Repository layout

```
index.html                 markup
assets/app.css             styles
assets/app.js              UI, image input, tiling preview, compare slider
assets/worker.js           ONNX Runtime Web, model cache, tiled inference
assets/models.js           model catalogue shared by both
coi-serviceworker.js       COOP/COEP shim for WASM threads
models/*.onnx              exported Real-ESRGAN weights (committed, served same-origin)
vendor/ort/                onnxruntime-web 1.27 runtime + WASM binary
tools/export_onnx.py       regenerates models/ from the upstream .pth checkpoints
tools/smoke-test.mjs       headless end-to-end test
```

There is no build step. Serve the directory over HTTP and it runs:

```sh
npx serve .          # or python3 -m http.server
```

### Regenerating the models

```sh
pip install torch onnx
mkdir weights && cd weights
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd .. && python tools/export_onnx.py --out models
```

The script re-implements SRVGGNetCompact and RRDBNet directly and loads the
checkpoints with `strict=True`, so a mismatch fails loudly instead of silently
producing garbage.

### Running the smoke test

```sh
npm install
node tools/smoke-test.mjs
```

It serves the directory, drives headless Chromium through a real upscale on
several models, and checks output dimensions, alpha handling and tile seams.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which uploads the
repository as-is to GitHub Pages. The workflow enables Pages on first run.

## Credits and licences

- Model weights: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al., BSD-3-Clause.
  The `.onnx` files here are format conversions of those published checkpoints.
- Inference: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), MIT.
- Everything else in this repository: MIT.
