"""
Generate the background-removal test fixtures with an exact ground truth.

    python tools/fixtures/make_bgremove_fixtures.py

Writes, next to this script:

    cutout-subject.png    the foreground F with its true alpha (RGBA)
    cutout-green.png      F composited over flat green #00b140
    cutout-texture.png    F composited over a busy, colourful texture

Because F, B and alpha are all known, the smoke test can measure mask IoU,
edge-band alpha error and how much green bleeds into the edges — exactly,
rather than against another model's opinion.

The subject is synthetic rather than a photograph so it is ours outright
(CC0): a shaded, furry toy-like character with a few hundred fine strands
around its outline, rendered at 4x and box-downsampled so thin hair gets a
genuinely fractional alpha. Deterministic: same seed, same pixels.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 640, 560
SS = 4                       # supersampling factor for coverage-based alpha
rng = np.random.default_rng(20261003)


def subject():
    """Return (rgb float [H,W,3], alpha float [H,W]) of the foreground."""
    w, h = W * SS, H * SS
    cov = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(cov)
    cx, cy = w * 0.5, h * 0.56
    rx, ry = w * 0.25, h * 0.30
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)                      # body
    for side in (-1, 1):                                                                # ears
        ex = cx + side * rx * 0.62
        draw.ellipse([ex - rx * 0.30, cy - ry * 1.28, ex + rx * 0.30, cy - ry * 0.62], fill=255)

    # Hair: curved strands leaving the outline, 1-2 px wide at output scale.
    hair = Image.new('L', (w, h), 0)
    hd = ImageDraw.Draw(hair)
    for _ in range(520):
        t = rng.uniform(0, 2 * np.pi)
        x0 = cx + np.cos(t) * rx * 0.97
        y0 = cy + np.sin(t) * ry * 0.97
        length = rng.uniform(0.10, 0.24) * rx
        bend = rng.uniform(-0.6, 0.6)
        pts = []
        for k in range(9):
            s = k / 8
            a = t + bend * s
            pts.append((x0 + np.cos(a) * length * s, y0 + np.sin(a) * length * s + 0.25 * length * s * s))
        hd.line(pts, fill=int(rng.uniform(150, 255)), width=int(rng.uniform(3, 7)))
    alpha_hi = np.maximum(np.asarray(cov, np.float32), np.asarray(hair, np.float32)) / 255

    # Colour: warm fur with shading and a darker belly patch; strands slightly lighter.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = np.clip(1.05 - 0.55 * (((xx - cx * 0.9) / rx) ** 2 + ((yy - cy * 0.8) / ry) ** 2), 0.45, 1.0)
    rgb = np.stack([0.93 * shade, 0.58 * shade, 0.28 * shade], -1)
    belly = (((xx - cx) / (rx * 0.55)) ** 2 + ((yy - cy - ry * 0.25) / (ry * 0.5)) ** 2) < 1
    rgb[belly] = rgb[belly] * np.array([1.0, 0.9, 0.8]) + 0.08
    strands = (np.asarray(hair) > 0) & (np.asarray(cov) == 0)
    rgb[strands] = np.array([0.98, 0.78, 0.48])

    # Down to output size: box filter, which turns coverage into fractional alpha.
    def down(a):
        return a.reshape(H, SS, W, SS, *a.shape[2:]).mean(axis=(1, 3))
    alpha = down(alpha_hi)
    # Premultiplied downsample so colour stays right where coverage is partial.
    rgb_pm = down(rgb * alpha_hi[..., None])
    rgb = np.where(alpha[..., None] > 0, rgb_pm / np.maximum(alpha[..., None], 1e-6), 0)

    # Eyes and nose drawn at output scale (fully inside the body, opaque).
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(img)
    ox, oy = cx / SS, cy / SS
    for side in (-1, 1):
        d.ellipse([ox + side * 38 - 13, oy - 46, ox + side * 38 + 13, oy - 18], fill=(250, 250, 245))
        d.ellipse([ox + side * 38 - 6, oy - 38, ox + side * 38 + 6, oy - 24], fill=(30, 22, 18))
    d.ellipse([ox - 10, oy - 8, ox + 10, oy + 6], fill=(60, 30, 25))
    rgb = np.asarray(img, np.float32) / 255
    return rgb, alpha


def texture():
    """A busy background: overlapping colourful shapes, stripes and noise."""
    img = Image.new('RGB', (W, H), (90, 120, 160))
    d = ImageDraw.Draw(img)
    for _ in range(140):
        x, y = rng.uniform(-60, W), rng.uniform(-60, H)
        s = rng.uniform(15, 110)
        colour = tuple(int(c) for c in rng.integers(20, 235, 3))
        if rng.random() < 0.5:
            d.ellipse([x, y, x + s, y + s], fill=colour)
        else:
            d.rectangle([x, y, x + s * 1.6, y + s * 0.5], fill=colour)
    # No thin light lines: at 1-2 px they would be indistinguishable from the
    # subject's hair strands, which tests camouflage rather than cutout quality.
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img, np.float32) / 255
    arr += rng.normal(0, 0.03, arr.shape).astype(np.float32)
    return np.clip(arr, 0, 1)


def save(arr, name, alpha=None):
    data = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    if alpha is not None:
        data = np.dstack([data, (np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)])
    Image.fromarray(data).save(os.path.join(HERE, name), optimize=True)
    print('wrote', name)


def main():
    F, alpha = subject()
    # Quantise once, then composite from the quantised values, so the PNGs on
    # disk are exactly F*a + B*(1-a) for the F and a the test reads back.
    Fq = np.round(F * 255) / 255
    aq = np.round(alpha * 255) / 255
    save(Fq, 'cutout-subject.png', aq)
    green = np.broadcast_to(np.array([0x00, 0xb1, 0x40], np.float32) / 255, F.shape)
    save(Fq * aq[..., None] + green * (1 - aq[..., None]), 'cutout-green.png')
    save(Fq * aq[..., None] + texture() * (1 - aq[..., None]), 'cutout-texture.png')
    edge = ((aq > 0) & (aq < 1)).mean()
    print(f'{W}x{H}, foreground {np.mean(aq > 0.5):.1%}, soft edge band {edge:.1%} of pixels')


if __name__ == '__main__':
    main()
