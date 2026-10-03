# drikus-upscale

AI image tools that run entirely in the browser, on your GPU (WebGPU) or CPU
(WebAssembly SIMD + threads). Nothing is uploaded — there is no server side.

- **Upscale** — drop in a small or JPEG-mangled picture, pick a model —
  [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) or
  [Swin2SR](https://github.com/mv-lab/swin2sr) — and it reconstructs the image at 4×.
- **Remove background** — an automatic cutout with a true soft alpha (hair, fur,
  motion blur) and colour-decontaminated edges, from
  [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) or [U²-Net](https://github.com/xuebinqin/U-2-Net).

The tools share one input, preview and download flow, and chain: *Use as input*
feeds any result back in, so you can cut out a subject and then upscale it, or
the other way round.

**→ https://drikusroor.github.io/drikus-upscale/**

## What it does

- **Three ways in**: drag and drop anywhere on the page, paste with <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>V</kbd>,
  or the regular file picker. Pasting an image URL works too when the host allows cross-origin reads.
- **A tool switch** (<kbd>U</kbd> / <kbd>R</kbd>), each tool with its own models and a recommended default (see below).
- **WebGPU when available, WASM otherwise**, decided automatically and overridable.
  If WebGPU fails to compile a model, the app falls back to WASM by itself.
- **Tiled inference in a worker**, so the UI stays responsive, memory stays bounded
  and the result paints in progressively, tile by tile.
- **Before/after slider**, PNG and WebP download, alpha channel preserved.
- **Models are cached** in the Cache API after the first download, so repeat runs
  are instant and the app works offline.

## Models

### Upscaling

Six are exported from the official Real-ESRGAN checkpoints; the seventh is
[Swin2SR](https://github.com/mv-lab/swin2sr) — a Swin transformer trained
specifically for compressed/JPEG-degraded inputs, a genuinely different
architecture from the GAN-based Real-ESRGAN family with different failure
modes. All output 4×; the "output size" selector resamples that down when you
want 3×, 2× or a same-size cleanup.

| Model | Size | Relative cost | Good for |
|---|---|---|---|
| **General ×4 — balanced** *(default)* | 4.9 MB | 1× | Photos, screenshots, JPEG artefacts — the all-rounder |
| General ×4 — max detail | 4.9 MB | 1× | Clean sources where you want every bit of texture |
| General ×4 — max denoise | 4.9 MB | 1× | Heavily compressed or noisy sources |
| Anime video ×4 — fast | 2.5 MB | 0.5× | Anime, flat-shaded art; the quickest option |
| Anime / art ×4 — 6B | 17.9 MB | 8× | Illustrations, manga scans, sprites |
| Photo ×4 — x4plus (heavy) | 67.1 MB | 30× | Best photo restoration, if you can wait |
| Photo ×4 — Swin2SR compressed (slowest) | 57.1 MB | 28× | JPEG artefacts the Real-ESRGAN models still leave visible |

The three "General" entries are the same `realesr-general-x4v3` network: the plain
weights, the WDN (denoise) weights, and — for the default — a 50/50 deep network
interpolation of the two, which is what `--denoise_strength 0.5` produces upstream.

Swin2SR is architecturally different in a way that matters for tiling: its
window-attention blocks bake an attention mask into the ONNX graph as a
constant, computed for one exact input resolution at export time. Feeding any
other resolution at inference would silently produce the wrong mask rather
than error, so — unlike the six CNN-based models, which tolerate any tile size
— it's exported for one fixed 96 px tile (128 px including its 16 px context
border), and the app locks the tile-size and overlap sliders to match whenever
it's selected. `tools/swin2sr/export.py` documents and reproduces the export.

### Rough upscaling speed

Time scales with input megapixels × the model's relative cost. On a 4-thread
WebAssembly build in CI, the general model does ~0.03 cost-weighted Mpx/s;
WebGPU is typically an order of magnitude faster. The app measures its own
throughput after each run and uses that for subsequent estimates, so the
"about N s" line gets accurate quickly.

### Background removal

| Model | Download | Input | Good for |
|---|---|---|---|
| **BiRefNet lite** *(default)* | 92 MB fp16 on WebGPU with `shader-f16`, otherwise 182 MB fp32 | 1024² | Clean edges on hair, fur and fine structure |
| U²-Netp — fast | 4.6 MB | 320² | Near-instant on any device; softer edges. Also the instant preview while BiRefNet runs, once cached |

The worker picks the weights from the device, not the user: fp16 on WebGPU
adapters with `shader-f16` (the capability chip shows *WebGPU · f16*), fp32 split
into two external-data shards everywhere else. Excluded on purpose: BRIA RMBG
(non-commercial weights), full BiRefNet (Swin-L, 444 MB in fp16) and any hosted
API.

After segmentation, everything runs at full resolution in the worker and
re-runs live (debounced, no model work) when a control moves:

```
image → 1024² squash → BiRefNet → S×S alpha ─┬→ IndexedDB mask cache
                                             └→ bilinear ↑ W×H · source α · threshold
                                                → colour guided filter (RGB guide, r, ε = 1e-3)
                                                → blur-fusion decontamination (r = 90, then 6)
                                                → RGBA strips → canvas
```

- **Edge softness** is the guided-filter radius (2–32 px). The guide is the full
  RGB image rather than luminance: it separates hair from a background of
  similar brightness, which on the green-screen fixture cuts edge-band alpha
  error by about a quarter. **Edge cleanup** is
  Forte & Pitié's blur-fusion foreground estimation, which removes background
  colour bleeding into hair and soft edges; **alpha threshold** cuts faint haze.
- **Strips.** Post-processing runs in horizontal strips with a margin equal to
  the sum of every downstream blur radius, so each strip matches a whole-image
  pass while memory stays bounded at any size — including 16384 px inputs
  chained from an upscale. Box filters use running sums: cost is O(pixels)
  whatever the radius.
- **Cache.** The raw mask is cached in IndexedDB, keyed by a SHA-256 of the
  decoded pixels plus model, variant and the exported graph's hash. The same
  image again — pasted or dropped — skips download, compile and inference
  entirely. LRU-evicted at 100 MB / 200 entries; "Clear cache" clears it.
- **Source alpha.** A source that is already transparent stays transparent: the
  predicted alpha is multiplied by it.
- **Output.** Transparent PNG/WebP (`_nobg`), the mask alone as an 8-bit
  greyscale PNG (`_mask`), or composited onto a solid colour.

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
assets/app.js              UI, tool switch, image input, preview, compare slider, chaining
assets/worker.js           ONNX Runtime Web, model cache, tiled upscaling, segmentation
assets/matting.js          full-res mask refinement + decontamination, in strips
assets/mask-cache.js       IndexedDB cache of raw segmentation masks
assets/png.js              8-bit greyscale PNG encoder for mask downloads
assets/models.js           tool and model catalogue shared by both
coi-serviceworker.js       COOP/COEP shim for WASM threads
models/*.onnx              exported model weights (committed, served same-origin)
vendor/ort/                onnxruntime-web 1.27 runtime + WASM binary
tools/export_onnx.py       regenerates the Real-ESRGAN models/ from the upstream .pth checkpoints
tools/swin2sr/export.py    regenerates the Swin2SR model from its upstream checkpoint
tools/swin2sr/vendor/      the upstream Swin2SR architecture file + a minimal timm shim
tools/bgremove/export.py   regenerates the background-removal models from their checkpoints
tools/bgremove/vendor/     upstream BiRefNet and U-2-Net architecture files + small shims
tools/fixtures/            test images; make_bgremove_fixtures.py generates the cutout ones
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

### Regenerating the Swin2SR model

```sh
mkdir -p weights && cd weights
curl -LO https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth
cd .. && python tools/swin2sr/export.py
```

Unlike the script above, this one uses the actual upstream architecture file
(`tools/swin2sr/vendor/network_swin2sr.py`, unmodified) rather than a
reimplementation, and traces a fixed input shape rather than a dynamic one —
see the module docstring in `export.py` for why that's required here.

### Regenerating the background-removal models

```sh
pip install torch torchvision onnx onnxruntime onnxslim onnxconverter-common einops numpy pillow
mkdir -p weights && cd weights
curl -LO https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth
curl -L -o u2netp-rembg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
cd .. && python tools/bgremove/export.py --verify-dir path/to/ten/images --check-deform
```

BiRefNet's decoder uses deformable convolutions, which ONNX has no op for and
which community exports turn into GatherND/ScatterND chains that exhaust ONNX
Runtime Web's memory. The script rewrites each one, before tracing, as a
GridSample per kernel tap plus a 1×1 convolution (`--check-deform` verifies the
rewrite against torchvision), keeps shapes static, bakes normalisation and the
sigmoid into the graph, deduplicates the initializers the backbone's two passes
share, and stores Swin's shifted-window masks as region ids rebuilt in-graph
instead of 17 MB of constants. It then checks the ONNX output against PyTorch
(max alpha difference ≤ 1e-3; fp16 against fp32 by mean difference and IoU) and
prints each file's size and sha256 for `assets/models.js`.

U²-Net's authors publish `u2netp.pth` only on Google Drive. If you have it, put it
in `weights/`; otherwise the script recovers the weights exactly from rembg's
ONNX conversion of the same checkpoint, which folded BatchNorm into the
convolutions, and checks the result against that graph.

### Running the smoke test

```sh
npm install
node tools/smoke-test.mjs
```

It serves the directory, drives headless Chromium through real upscales and
background removals, and checks output dimensions, alpha handling and tile
seams; cutout IoU, edge-band alpha error and edge-colour decontamination against
synthetic fixtures with an exact ground truth; that a repeat run is a cache hit
and that refining never re-runs the model; remove → upscale chaining; and
cancellation. `ONLY=upscale` or `ONLY=remove` runs one tool's cases.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which uploads the
repository as-is to GitHub Pages. The workflow enables Pages on first run.

## Credits and licences

- Model weights: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al., BSD-3-Clause;
  [Swin2SR](https://github.com/mv-lab/swin2sr) by Conde, Choi, Burchi and Timofte, Apache-2.0;
  [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by Zheng Peng et al., MIT;
  and [U²-Net](https://github.com/xuebinqin/U-2-Net) by Xuebin Qin et al., Apache-2.0.
  The `.onnx` files here are format conversions of those published checkpoints (see `models/LICENSE.md`).
- Inference: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), MIT.
- Everything else in this repository: MIT.
