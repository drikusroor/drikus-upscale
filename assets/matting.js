/*
 * Full-resolution matte post-processing for background removal.
 *
 * The segmentation model produces one S×S alpha (S = 1024 or 320). Everything
 * here turns that into a W×H cutout and is cheap enough to re-run live when a
 * refine slider moves:
 *
 *   upsample  bilinear S×S → W×H, × source alpha, threshold
 *   refine    colour guided filter (He et al.) with the full-res image as the guide
 *   decontam  blur-fusion foreground estimation (Forte & Pitié, 2021), r = 90 then 6
 *   pack      RGBA with RGB = 0 wherever α = 0
 *
 * All of it is box filters, so it runs in horizontal strips: each strip is
 * computed over a margin equal to the sum of every downstream blur radius,
 * which makes it match a whole-image pass (up to float rounding) while
 * keeping memory bounded at any image size. Box filters use running sums, so
 * cost is O(pixels) whatever the radius.
 *
 * Pure functions over typed arrays, no DOM: the worker uses them, and so can
 * a Node test.
 */

export const DEFAULT_REFINE = { radius: 8, eps: 1e-3, threshold: 0, decontaminate: true };
export const FUSION_RADII = [90, 6];

/** Extra rows above and below a strip so that its output is exact. */
export function stripMargin(refine) {
  const fusion = refine.decontaminate ? FUSION_RADII.reduce((a, b) => a + b, 0) : 0;
  return 2 * refine.radius + fusion;
}

/**
 * Rows per strip. 512 by default, smaller for very wide images so the working
 * set (fourteen float planes over strip + 2 × margin rows) stays near
 * `budgetPx` pixels — about 225 MB at the default.
 */
export function stripRows(width, margin, budgetPx = 4_000_000) {
  const regionRows = Math.floor(budgetPx / Math.max(1, width));
  return Math.max(16, Math.min(512, regionRows - 2 * margin));
}

/* ----------------------------------------------------------------- box filters */

/**
 * In-place horizontal box mean of radius r over each of `rows` rows of width
 * w. Windows are clipped at the image edge and divided by their in-bounds
 * count; the interior loop has a constant divisor and no branches.
 */
function boxH(plane, w, rows, r, line) {
  if (2 * r + 1 > w) return boxHNarrow(plane, w, rows, r, line);
  const inv = 1 / (2 * r + 1);
  const end = w - r - 1;
  for (let y = 0; y < rows; y++) {
    const o = y * w;
    line.set(plane.subarray(o, o + w));
    let acc = 0;
    for (let x = 0; x <= r; x++) acc += line[x];
    for (let x = 0; x < r; x++) { plane[o + x] = acc / (x + r + 1); acc += line[x + r + 1]; }
    for (let x = r; x < end; x++) { plane[o + x] = acc * inv; acc += line[x + r + 1] - line[x - r]; }
    for (let x = end; x < w; x++) { plane[o + x] = acc / (w - x + r); acc -= line[x - r]; }
  }
}

/** The same, for rows narrower than one window. */
function boxHNarrow(plane, w, rows, r, line) {
  for (let y = 0; y < rows; y++) {
    const o = y * w;
    line.set(plane.subarray(o, o + w));
    let acc = 0;
    for (let x = 0; x < w; x++) acc += line[x];
    const mean = acc / w;
    for (let x = 0; x < w; x++) {
      const lo = Math.max(0, x - r);
      const hi = Math.min(w - 1, x + r);
      if (lo === 0 && hi === w - 1) { plane[o + x] = mean; continue; }
      let sum = 0;
      for (let k = lo; k <= hi; k++) sum += line[k];
      plane[o + x] = sum / (hi - lo + 1);
    }
  }
}

/**
 * In-place vertical box mean. Rows leaving the window have already been
 * overwritten, so the last r + 1 originals live in a ring buffer.
 */
function boxV(plane, w, rows, r, acc, ring) {
  const slots = r + 1;
  acc.fill(0);
  const first = Math.min(r, rows - 1);
  for (let y = 0; y <= first; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) acc[x] += plane[o + x];
  }
  for (let y = 0; y < rows; y++) {
    const o = y * w;
    const lo = y - r < 0 ? 0 : y - r;
    const hi = y + r > rows - 1 ? rows - 1 : y + r;
    const inv = 1 / (hi - lo + 1);
    const slot = (y % slots) * w;
    for (let x = 0; x < w; x++) {
      ring[slot + x] = plane[o + x];
      plane[o + x] = acc[x] * inv;
    }
    if (y + r + 1 < rows) {
      const add = (y + r + 1) * w;
      for (let x = 0; x < w; x++) acc[x] += plane[add + x];
    }
    if (y - r >= 0) {
      const sub = ((y - r) % slots) * w;
      for (let x = 0; x < w; x++) acc[x] -= ring[sub + x];
    }
  }
}

/** Scratch buffers for box filtering, reused across strips and calls. */
export class Scratch {
  constructor() { this.cap = 0; this.ringCap = 0; this.planes = []; }

  ensure(width, rows, maxRadius, planeCount) {
    if (!this.line || this.line.length < width) {
      this.line = new Float32Array(width);
      this.acc = new Float64Array(width);
    }
    const ring = (maxRadius + 1) * width;
    if (ring > this.ringCap) { this.ring = new Float32Array(ring); this.ringCap = ring; }
    const size = width * rows;
    if (size > this.cap || this.planes.length < planeCount) {
      this.planes = Array.from({ length: planeCount }, () => new Float32Array(size));
      this.cap = size;
    }
    return this.planes.map((p) => p.subarray(0, size));
  }

  /** Drop the buffers (up to a couple of hundred MB) once no image is held. */
  release() {
    this.cap = 0;
    this.ringCap = 0;
    this.planes = [];
    this.ring = null;
    this.line = null;
  }
}

export function box(plane, w, rows, r, scratch) {
  if (r <= 0) return;
  boxH(plane, w, rows, r, scratch.line);
  boxV(plane, w, rows, r, scratch.acc, scratch.ring);
}

/* ----------------------------------------------------------------- stages */

/**
 * Bilinear upsample of the S×S mask into rows [y0, y0 + rows) of a W×H frame,
 * the same squash the model was fed (no letterbox), times the source alpha,
 * then the threshold: values below t go to zero and the rest stretch to [0, 1].
 */
function upsampleRows(mask, size, width, height, y0, rows, src, threshold, out) {
  const sx = size / width;
  const sy = size / height;
  const keep = threshold > 0 ? 1 / (1 - threshold) : 1;
  for (let y = 0; y < rows; y++) {
    let fy = (y0 + y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    const iy = Math.min(size - 1, Math.floor(fy));
    const iy1 = Math.min(size - 1, iy + 1);
    const wy = fy - iy;
    const r0 = iy * size;
    const r1 = iy1 * size;
    const so = (y0 + y) * width;
    const o = y * width;
    for (let x = 0; x < width; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      const ix = Math.min(size - 1, Math.floor(fx));
      const ix1 = Math.min(size - 1, ix + 1);
      const wx = fx - ix;
      const top = mask[r0 + ix] + (mask[r0 + ix1] - mask[r0 + ix]) * wx;
      const bot = mask[r1 + ix] + (mask[r1 + ix1] - mask[r1 + ix]) * wx;
      let a = ((top + (bot - top) * wy) / 255) * (src[(so + x) * 4 + 3] / 255);
      if (threshold > 0) a = a < threshold ? 0 : (a - threshold) * keep;
      out[o + x] = a;
    }
  }
}

/**
 * Colour guided filter (He et al., 2013), in place on `p`, with the source RGB
 * as the guide. A colour guide separates hair from a background of similar
 * brightness that a luminance guide cannot (on the green-screen fixture it
 * cuts edge-band alpha error by about a quarter). Thirteen box-filtered
 * planes and a per-pixel 3×3 solve, all O(pixels).
 */
function guidedFilter(src, width, ry0, rows, r, eps, p, scratch, P) {
  const [mr, mg, mb, mp, pr, pg, pb, rr, rg, rb, gg, gb, bb] = P;
  const n = width * rows;
  const k = 1 / 255;
  for (let i = 0, s = ry0 * width * 4; i < n; i++, s += 4) {
    const R = src[s] * k, G = src[s + 1] * k, B = src[s + 2] * k, q = p[i];
    mr[i] = R; mg[i] = G; mb[i] = B; mp[i] = q;
    pr[i] = R * q; pg[i] = G * q; pb[i] = B * q;
    rr[i] = R * R; rg[i] = R * G; rb[i] = R * B; gg[i] = G * G; gb[i] = G * B; bb[i] = B * B;
  }
  for (const plane of P) box(plane, width, rows, r, scratch);
  for (let i = 0; i < n; i++) {
    const ar = mr[i], ag = mg[i], ab = mb[i], am = mp[i];
    // Σ + εI, symmetric, and its adjugate.
    const vrr = rr[i] - ar * ar + eps, vrg = rg[i] - ar * ag, vrb = rb[i] - ar * ab;
    const vgg = gg[i] - ag * ag + eps, vgb = gb[i] - ag * ab, vbb = bb[i] - ab * ab + eps;
    const cr = pr[i] - ar * am, cg = pg[i] - ag * am, cb = pb[i] - ab * am;
    const i00 = vgg * vbb - vgb * vgb, i01 = vrb * vgb - vrg * vbb, i02 = vrg * vgb - vrb * vgg;
    const i11 = vrr * vbb - vrb * vrb, i12 = vrg * vrb - vrr * vgb, i22 = vrr * vgg - vrg * vrg;
    const inv = 1 / (vrr * i00 + vrg * i01 + vrb * i02);
    const xr = (i00 * cr + i01 * cg + i02 * cb) * inv;
    const xg = (i01 * cr + i11 * cg + i12 * cb) * inv;
    const xb = (i02 * cr + i12 * cg + i22 * cb) * inv;
    pr[i] = xr; pg[i] = xg; pb[i] = xb;
    mp[i] = am - xr * ar - xg * ag - xb * ab;
  }
  box(pr, width, rows, r, scratch);
  box(pg, width, rows, r, scratch);
  box(pb, width, rows, r, scratch);
  box(mp, width, rows, r, scratch);
  for (let i = 0, s = ry0 * width * 4; i < n; i++, s += 4) {
    const q = pr[i] * src[s] * k + pg[i] * src[s + 1] * k + pb[i] * src[s + 2] * k + mp[i];
    p[i] = q < 0 ? 0 : q > 1 ? 1 : q;
  }
}

/**
 * One blur-fusion pass, channel by channel, in place on F and B:
 *   F̂ = blur(F α) / blur(α),  B̂ = blur(B (1 − α)) / blur(1 − α)
 *   F ← clip(F̂ + α (I − α F̂ − (1 − α) B̂)),  B ← B̂
 * With mean-normalised box filters blur(1 − α) = 1 − blur(α).
 */
function blurFusion(src, width, y0, rows, alpha, F, B, r, scratch, [bA, X, Y]) {
  const n = width * rows;
  bA.set(alpha);
  box(bA, width, rows, r, scratch);
  for (let c = 0; c < 3; c++) {
    const Fc = F[c];
    const Bc = B[c];
    for (let i = 0; i < n; i++) {
      X[i] = Fc[i] * alpha[i];
      Y[i] = Bc[i] * (1 - alpha[i]);
    }
    box(X, width, rows, r, scratch);
    box(Y, width, rows, r, scratch);
    for (let i = 0, s = y0 * width * 4 + c; i < n; i++, s += 4) {
      const a = alpha[i];
      const f = X[i] / (bA[i] + 1e-5);
      const b = Y[i] / (1 - bA[i] + 1e-5);
      const v = f + a * (src[s] / 255 - a * f - (1 - a) * b);
      Fc[i] = v < 0 ? 0 : v > 1 ? 1 : v;
      Bc[i] = b;
    }
  }
}

/**
 * Produce output rows [y0, y1) of the cutout as RGBA.
 *
 * `src` is the full-resolution source RGBA, `mask` the model's S×S uint8
 * alpha. Returns a Uint8ClampedArray of (y1 − y0) × width × 4.
 */
export function processStrip({ src, width, height, mask, size, refine, y0, y1, scratch }) {
  const margin = stripMargin(refine);
  const ry0 = Math.max(0, y0 - margin);
  const ry1 = Math.min(height, y1 + margin);
  const rows = ry1 - ry0;
  const radius = refine.radius;
  const maxR = Math.max(radius, refine.decontaminate ? FUSION_RADII[0] : 0);
  const planes = scratch.ensure(width, rows, maxR, 14);
  const [alpha, ...P] = planes;           // P: 13 working planes, shared by both stages

  upsampleRows(mask, size, width, height, ry0, rows, src, refine.threshold, alpha);
  if (radius > 0) guidedFilter(src, width, ry0, rows, radius, refine.eps, alpha, scratch, P);
  // Re-apply the source alpha: the guided filter must not bring back pixels
  // that were transparent in the source (FR-3), and anything under half a
  // grey level is background noise rather than matte.
  const n = width * rows;
  for (let i = 0, s = ry0 * width * 4 + 3; i < n; i++, s += 4) {
    const a = alpha[i] * (src[s] / 255);
    alpha[i] = a < 0.002 ? 0 : a;
  }

  const [F0, F1, F2, B0, B1, B2, s0, s1, s2] = P;
  const F = [F0, F1, F2];
  // Decontamination only changes pixels with fractional alpha: at α = 1 the
  // update reduces to F = I, and α = 0 is written as RGB 0. A strip whose own
  // rows are all hard 0/1 (solid subject or plain background) skips it exactly.
  const outStart = (y0 - ry0) * width;
  const outEnd = outStart + (y1 - y0) * width;
  let soft = false;
  for (let i = outStart; i < outEnd; i++) {
    if (alpha[i] > 0 && alpha[i] < 1) { soft = true; break; }
  }
  const decontaminate = refine.decontaminate && soft;
  if (decontaminate) {
    const B = [B0, B1, B2];
    for (let c = 0; c < 3; c++) {
      for (let i = 0, s = ry0 * width * 4 + c; i < n; i++, s += 4) {
        F[c][i] = B[c][i] = src[s] / 255;
      }
    }
    for (const r of FUSION_RADII) blurFusion(src, width, ry0, rows, alpha, F, B, r, scratch, [s0, s1, s2]);
  }

  const outRows = y1 - y0;
  const out = new Uint8ClampedArray(outRows * width * 4);
  const off = (y0 - ry0) * width;
  for (let i = 0, n2 = outRows * width; i < n2; i++) {
    const a = alpha[off + i];
    const d = i * 4;
    if (a <= 0) continue;                      // RGB 0 under α 0 — smaller PNGs
    if (decontaminate) {
      out[d] = F0[off + i] * 255 + 0.5;
      out[d + 1] = F1[off + i] * 255 + 0.5;
      out[d + 2] = F2[off + i] * 255 + 0.5;
    } else {
      const s = ((y0 * width) + i) * 4;
      out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2];
    }
    out[d + 3] = a * 255 + 0.5;
  }
  return out;
}
