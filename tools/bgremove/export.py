"""
Export the background-removal models to browser-ready ONNX graphs.

Writes, into models/:

    birefnet-lite.fp32.onnx  + birefnet-lite.fp32.data0/.data1   (WASM, WebGPU without shader-f16)
    birefnet-lite.fp16.onnx                                       (WebGPU with shader-f16)
    u2netp.onnx                                                    (both backends)

Every graph satisfies the same contract, so the worker needs no per-model
pre- or post-processing code:

    input   `input`   float32 [1, 3, S, S], RGB in [0, 1]   (S = 1024 / 320)
    output  `output`  float32 [1, 1, S, S], alpha in [0, 1]
    ImageNet normalisation, sigmoid and a final Clip(0, 1) live inside the graph.

Why this script exists rather than using a community ONNX file as-is:

  * BiRefNet's decoder uses torchvision's deformable convolution, which has no
    ONNX op. Exporters fall back to GatherND/ScatterND chains that blow up
    ONNX Runtime Web's memory (`std::bad_alloc`). Here every DeformableConv2d
    is rewritten before tracing as one GridSample per kernel tap followed by a
    1x1 Conv, summed with Add (not Sum, which the WebGPU EP lacks for fp16).
    `--check-deform` verifies the rewrite against torchvision numerically.
  * Shapes are static. Swin window attention bakes its padding and shift mask
    at trace time, exactly as Swin2SR does (see tools/swin2sr/export.py).
  * BiRefNet runs its backbone twice (full and half resolution), and constant
    folding produces identical copies of the relative-position tables for each
    pass. Initializers are deduplicated by content hash, which is what keeps
    the fp16 file under GitHub's 100 MB per-file limit.
  * The fp32 weights (~178 MB) are written as ONNX external data in two shards,
    each under 100 MB, which ONNX Runtime Web loads through `externalData`.

Weights:

  * BiRefNet_lite (MIT): the official release checkpoint
        https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth
  * U^2-Netp (Apache-2.0): the official `u2netp.pth` if you have it (upstream
    only publishes it on Google Drive). Otherwise the weights are recovered
    exactly from rembg's ONNX export of the same checkpoint
        https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
    That export folded each BatchNorm into its Conv; the recovery assigns the
    folded Conv weights and sets every BatchNorm to the identity, which is
    numerically the same network. The state dict is still loaded strictly and
    the result is checked against rembg's graph output.

Usage:
    pip install torch torchvision onnx onnxruntime onnxslim onnxconverter-common einops numpy pillow
    mkdir -p weights && cd weights
    curl -LO https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth
    curl -L -o u2netp-rembg.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
    cd .. && python tools/bgremove/export.py --verify-dir path/to/10/images

The architectures are the upstream files under vendor/, unmodified, with small
stand-ins for their training-only imports (see each shim's docstring).
"""
import argparse
import glob
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
WEIGHTS = os.path.join(ROOT, 'weights')
OUT = os.path.join(ROOT, 'models')

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SHARD_LIMIT = 99_000_000      # bytes per external-data shard, under GitHub's 100 MB


# --------------------------------------------------------------------------- deformable conv rewrite

def deform_conv_as_grid_sample(self, x):
    """DeformableConv2d.forward, rewritten with ops ONNX Runtime Web runs everywhere.

    torchvision's modulated deformable convolution samples, for each kernel tap
    t = (i, j), the input at p + (i - pad, j - pad) + offset_t(p), bilinearly
    with zeros outside the image, scales it by mask_t(p) and applies the tap's
    slice of the weight. GridSample with align_corners=True reproduces that
    sampling exactly (pixel centres at integer coordinates, zero padding), and
    the per-tap weight slice is a 1x1 convolution. Offsets are laid out
    (dy, dx) per tap, matching torchvision's kernel.

    Slices are taken one tap at a time rather than with a single Split, since
    the 7x7 branch would need a 49-way Split and Dawn binds at most 16 storage
    buffers per shader.
    """
    assert self.stride == (1, 1)
    offset = self.offset_conv(x)
    modulator = 2. * torch.sigmoid(self.modulator_conv(x))
    weight = self.regular_conv.weight
    # int(): under tracing, .shape entries are tensors; these must be Python
    # constants so the loop unrolls and the sampling grid folds to a constant.
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    h, w = int(x.shape[2]), int(x.shape[3])
    pad = self.padding if isinstance(self.padding, int) else self.padding[0]

    ys = torch.arange(h, dtype=x.dtype).view(1, h, 1)
    xs = torch.arange(w, dtype=x.dtype).view(1, 1, w)
    sy = 2.0 / max(h - 1, 1)
    sx = 2.0 / max(w - 1, 1)

    out = None
    for t in range(kh * kw):
        i, j = divmod(t, kw)
        dy = offset[:, 2 * t]
        dx = offset[:, 2 * t + 1]
        gy = (ys + (i - pad) + dy) * sy - 1.0
        gx = (xs + (j - pad) + dx) * sx - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled * modulator[:, t:t + 1]
        term = F.conv2d(sampled, weight[:, :, i:i + 1, j:j + 1])
        out = term if out is None else out + term
    if self.regular_conv.bias is not None:
        out = out + self.regular_conv.bias.view(1, -1, 1, 1)
    return out


# --------------------------------------------------------------------------- wrappers

class Segmenter(nn.Module):
    """Bakes the app's contract around a segmentation network: [0,1] RGB in,
    ImageNet-normalised internally, sigmoid alpha out, clipped to [0, 1]."""

    def __init__(self, net, kind):
        super().__init__()
        self.net = net
        self.kind = kind
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        if self.kind == 'u2netp':
            # Upstream U^2-Net inference divides by the image maximum before
            # normalising, then min-max stretches the first side output.
            x = x / torch.clamp(x.amax(dim=(1, 2, 3), keepdim=True), min=1e-6)
            d = self.net((x - self.mean) / self.std)[0]       # already sigmoid
            lo = d.amin(dim=(1, 2, 3), keepdim=True)
            hi = d.amax(dim=(1, 2, 3), keepdim=True)
            alpha = (d - lo) / torch.clamp(hi - lo, min=1e-6)
        else:
            alpha = torch.sigmoid(self.net((x - self.mean) / self.std)[-1])
        return torch.clamp(alpha, 0.0, 1.0)


# --------------------------------------------------------------------------- model builders

def build_birefnet():
    sys.path.insert(0, os.path.join(HERE, 'vendor', 'birefnet'))
    from models.birefnet import BiRefNet
    net = BiRefNet(bb_pretrained=False)
    path = os.path.join(WEIGHTS, 'BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth')
    state = torch.load(path, map_location='cpu', weights_only=True)
    # The release checkpoint predates upstream renaming `squeeze_0` to
    # `squeeze_module.0`; that is the only difference from the current code.
    state = {k.replace('squeeze_0.', 'squeeze_module.0.'): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return net.eval()


def build_u2netp():
    sys.path.insert(0, os.path.join(HERE, 'vendor', 'u2net'))
    from u2net import U2NETP
    net = U2NETP(3, 1).eval()
    pth = os.path.join(WEIGHTS, 'u2netp.pth')
    if os.path.exists(pth):
        net.load_state_dict(torch.load(pth, map_location='cpu', weights_only=True), strict=True)
        return net, None
    rembg = os.path.join(WEIGHTS, 'u2netp-rembg.onnx')
    net.load_state_dict(recover_u2netp_state(net, rembg), strict=True)
    return net, rembg


def recover_u2netp_state(net, onnx_path):
    """Rebuild a U2NETP state dict from rembg's BN-folded ONNX export.

    torch.onnx emitted the Conv nodes in execution order, so they pair one to
    one with the Conv2d modules in the order a forward pass calls them. Each
    pairing is checked on shape and dilation before anything is assigned."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(onnx_path).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    convs = [n for n in graph.node if n.op_type == 'Conv']

    order = []
    hooks = [m.register_forward_hook(lambda mod, _i, _o: order.append(mod))
             for m in net.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        net(torch.zeros(1, 3, 320, 320))
    for h in hooks:
        h.remove()
    if len(order) != len(convs):
        raise SystemExit(f'u2netp: {len(order)} convs in PyTorch vs {len(convs)} in ONNX')

    names = {m: n for n, m in net.named_modules()}
    state = {k: v.clone() for k, v in net.state_dict().items()}
    for module, node in zip(order, convs):
        name = names[module]
        weight = inits[node.input[1]]
        bias = inits[node.input[2]] if len(node.input) > 2 else np.zeros(weight.shape[0], np.float32)
        dil = next((list(a.ints) for a in node.attribute if a.name == 'dilations'), [1, 1])
        if tuple(weight.shape) != tuple(module.weight.shape) or tuple(dil) != tuple(module.dilation):
            raise SystemExit(f'u2netp: conv {name} does not match ONNX node {node.name}')
        state[f'{name}.weight'] = torch.from_numpy(weight.copy())
        state[f'{name}.bias'] = torch.from_numpy(bias.copy())
        bn = name.rsplit('.', 1)[0] + '.bn_s1'
        if f'{bn}.weight' in state:
            c = weight.shape[0]
            eps = dict(net.named_modules())[bn].eps
            state[f'{bn}.weight'] = torch.ones(c)
            state[f'{bn}.bias'] = torch.zeros(c)
            state[f'{bn}.running_mean'] = torch.zeros(c)
            state[f'{bn}.running_var'] = torch.full((c,), 1.0 - eps)
    return state


# --------------------------------------------------------------------------- graph post-processing

def dedupe_initializers(model):
    """Point every node at one copy of each distinct initializer."""
    import onnx
    from onnx import numpy_helper
    seen, rename, keep = {}, {}, []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        key = (init.data_type, tuple(init.dims), hashlib.sha256(arr.tobytes()).hexdigest())
        if key in seen:
            rename[init.name] = seen[key]
        else:
            seen[key] = init.name
            keep.append(init)
    if not rename:
        return model, 0
    for node in model.graph.node:
        for k, name in enumerate(node.input):
            if name in rename:
                node.input[k] = rename[name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return model, len(rename)


def compact_attention_masks(model):
    """Swap Swin's baked shifted-window masks for a tiny equivalent subgraph.

    Each shifted block adds a constant [1, nW, 1, N, N] mask that is 0 where
    two tokens of a window come from the same pre-shift region and -inf
    elsewhere. At 1024 px the first stage alone has 1369 windows, so the masks
    cost ~17 MB of fp32 for what is really one region id per token. This
    stores those ids ([1, nW, 1, N, 1]) and rebuilds the mask in-graph as
    Where(ids == ids^T, 0, -inf), which is exactly the same tensor.
    """
    import onnx
    from onnx import helper, numpy_helper
    saved = 0
    for init in list(model.graph.initializer):
        dims = tuple(init.dims)
        if len(dims) != 5 or dims[0] != 1 or dims[2] != 1 or dims[3] != dims[4] or np.prod(dims) < 100_000:
            continue
        mask = numpy_helper.to_array(init)
        blocked = ~(mask == 0)
        if not np.all(np.isneginf(mask[blocked]) | (mask[blocked] <= -100)):
            continue
        fill = float(mask[blocked].max()) if blocked.any() else float('-inf')
        same = ~blocked[0, :, 0]                                   # [nW, N, N]
        ids = same.argmax(axis=2).astype(np.float32)               # first token in the same region
        rebuilt = ids[:, :, None] == ids[:, None, :]
        if not np.array_equal(rebuilt, same):
            continue                                               # not a region mask; leave it be
        name = init.name
        prefix = name.replace('/', '_').strip('_')
        model.graph.initializer.remove(init)
        model.graph.initializer.extend([
            numpy_helper.from_array(ids.reshape(1, dims[1], 1, dims[3], 1), f'{prefix}_region_ids'),
            numpy_helper.from_array(np.array(0, np.float32), f'{prefix}_zero'),
            numpy_helper.from_array(np.array(fill, np.float32), f'{prefix}_fill'),
        ])
        nodes = [
            helper.make_node('Transpose', [f'{prefix}_region_ids'], [f'{prefix}_region_ids_t'], perm=[0, 1, 2, 4, 3]),
            helper.make_node('Equal', [f'{prefix}_region_ids', f'{prefix}_region_ids_t'], [f'{prefix}_same']),
            helper.make_node('Where', [f'{prefix}_same', f'{prefix}_zero', f'{prefix}_fill'], [name]),
        ]
        for k, node in enumerate(nodes):
            model.graph.node.insert(k, node)
        saved += mask.nbytes - ids.nbytes
    return model, saved


def name_output_dims(model):
    """Name the output axes height_out/width_out (never the input's names).

    The shapes are static, so this asserts nothing about equality with the
    input; it simply follows the convention every graph in models/ uses so a
    future dynamic re-export can't regress into the buffer-reuse bug the
    README describes."""
    dims = model.graph.output[0].type.tensor_type.shape.dim
    dims[2].dim_param = 'height_out'
    dims[3].dim_param = 'width_out'
    return model


def finalize(model):
    import onnxslim
    # Constant folding + dead-code removal. Gemm fusion stays off: the backbone
    # runs twice, and fusing only one pass's MatMul+Add into Gemm stores that
    # pass's weights in the other orientation, which defeats deduplication
    # (about 100 MB of transposed copies).
    model = onnxslim.slim(model, skip_fusion_patterns=['FusionGemm'])
    model, dropped = dedupe_initializers(model)
    model, saved = compact_attention_masks(model)
    model = name_output_dims(model)
    print(f'  deduplicated {dropped} initializers; attention masks compacted by {saved / 1e6:.1f} MB')
    return model


def check_webgpu_placement(model):
    """Static checks for the WebGPU EP rules in the spec."""
    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    bad = [n.name for n in model.graph.node
           if n.op_type == 'Split' and len(n.output) > 15]
    for op in ('GatherND', 'ScatterND', 'Sum'):
        if ops.get(op):
            bad.append(f'{ops[op]}x {op}')
    if bad:
        raise SystemExit(f'graph violates the WebGPU contract: {bad}')
    top = ', '.join(f'{k} {v}' for k, v in ops.most_common(8))
    print(f'  {sum(ops.values())} nodes ({top})')


def save_sharded(model, path, base, shards=2):
    """Save with external data in `shards` files, each under SHARD_LIMIT bytes.

    Tensors are packed first-fit-decreasing, so the shards fill evenly instead
    of spilling a sliver into an extra file."""
    import onnx
    from onnx.external_data_helper import set_external_data
    big = sorted((t for t in model.graph.initializer if len(t.raw_data) >= 1024),
                 key=lambda t: len(t.raw_data), reverse=True)
    names = [f'{base}.data{i}' for i in range(shards)]
    fill = [0] * shards
    handles = [open(os.path.join(os.path.dirname(path), n), 'wb') for n in names]
    try:
        for tensor in big:
            raw = tensor.raw_data
            k = next((i for i in range(shards) if fill[i] + len(raw) <= SHARD_LIMIT), None)
            if k is None:
                raise SystemExit(f'weights do not fit in {shards} shards of {SHARD_LIMIT:,} bytes')
            set_external_data(tensor, location=names[k], offset=fill[k], length=len(raw))
            handles[k].write(raw)
            fill[k] += len(raw)
            tensor.ClearField('raw_data')
            tensor.data_location = onnx.TensorProto.EXTERNAL
    finally:
        for h in handles:
            h.close()
    onnx.save_model(model, path)
    return names


def to_fp16(model):
    from onnxconverter_common import float16
    # Inputs and outputs stay float32 so the worker code is identical across
    # variants. LayerNormalization and Softmax stay fp32: Swin's pre-norm
    # residual stream exceeds fp16's range on some inputs at 1024 px.
    return float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True,
        op_block_list=['LayerNormalization', 'Softmax', 'ReduceMean'])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def report(paths):
    for p in paths:
        print(f'  {os.path.basename(p)}  {os.path.getsize(p):>11,d} B  sha256 {sha256(p)}')
        if os.path.getsize(p) >= 100_000_000:
            raise SystemExit(f'{p} is over GitHub Pages\' 100 MB per-file limit')


# --------------------------------------------------------------------------- verification

def load_images(directory, size, count=10):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(directory, '*')))[:count] if directory else []
    images = []
    for f in files:
        try:
            im = Image.open(f).convert('RGB').resize((size, size), Image.BILINEAR)
        except Exception:
            continue
        images.append(torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None])
    rng = np.random.default_rng(7)
    while len(images) < count:     # pad with smooth synthetic scenes if fewer images were given
        base = rng.random((3, 8, 8)).astype(np.float32)
        images.append(F.interpolate(torch.from_numpy(base)[None], size=(size, size), mode='bicubic').clamp(0, 1))
    return images


def verify(onnx_path, reference, images, limit, label):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    worst = 0.0
    for img in images:
        with torch.no_grad():
            ref = reference(img).numpy()
        got = sess.run(['output'], {'input': img.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    ok = worst <= limit
    print(f'  {label}: max |alpha - reference| over {len(images)} images = {worst:.2e}  ({"PASS" if ok else "FAIL"}, limit {limit:g})')
    if not ok:
        raise SystemExit(f'{label} verification failed')


def verify_fp16(fp16_path, fp32_path, images):
    """fp16 can't match to 1e-3 per pixel (a handful of edge pixels move by a
    few percent), so it is held to the masks it produces instead: mean
    absolute difference and IoU at 0.5 against the fp32 graph."""
    import onnxruntime as ort
    a = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(fp16_path, providers=['CPUExecutionProvider'])
    worst_mean, worst_iou = 0.0, 1.0
    for img in images:
        p = a.run(['output'], {'input': img.numpy()})[0]
        q = b.run(['output'], {'input': img.numpy()})[0]
        worst_mean = max(worst_mean, float(np.abs(p - q).mean()))
        union = ((p > .5) | (q > .5)).sum()
        worst_iou = min(worst_iou, float(((p > .5) & (q > .5)).sum() / union) if union else 1.0)
    ok = worst_mean <= 2e-3 and worst_iou >= 0.99
    print(f'  fp16 vs fp32 over {len(images)} images: worst mean |diff| {worst_mean:.1e}, '
          f'worst IoU {worst_iou:.4f}  ({"PASS" if ok else "FAIL"}, limits 2e-3 / 0.99)')
    if not ok:
        raise SystemExit('fp16 verification failed')


# --------------------------------------------------------------------------- main

def export(module, size, path):
    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(
        module, dummy, path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes=None, dynamo=False,
    )


def main():
    import onnx
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--only', choices=['birefnet-lite', 'u2netp'])
    parser.add_argument('--verify-dir', help='directory with (at least) 10 test images')
    parser.add_argument('--check-deform', action='store_true',
                        help='also compare the GridSample rewrite against torchvision before exporting')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(WEIGHTS, 'tmp-export.onnx')

    if args.only in (None, 'u2netp'):
        print('U^2-Netp (320 px)')
        net, rembg = build_u2netp()
        model = Segmenter(net, 'u2netp').eval()
        path = os.path.join(OUT, 'u2netp.onnx')
        export(model, 320, tmp)
        m = finalize(onnx.load(tmp))
        check_webgpu_placement(m)
        onnx.save_model(m, path)
        images = load_images(args.verify_dir, 320)
        verify(path, model, images, 1e-3, 'onnx vs pytorch')
        if rembg:
            import onnxruntime as ort
            ref = ort.InferenceSession(rembg, providers=['CPUExecutionProvider'])
            worst = 0.0
            for img in images:
                x = img / img.amax().clamp(min=1e-6)
                x = ((x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)).numpy()
                with torch.no_grad():
                    ours = net(torch.from_numpy(x))[0].numpy()
                theirs = ref.run(None, {ref.get_inputs()[0].name: x})[0]
                worst = max(worst, float(np.abs(ours - theirs).max()))
            print(f'  recovered weights vs rembg graph: max diff {worst:.2e}')
            if worst > 1e-4:
                raise SystemExit('recovered U^2-Netp weights do not reproduce the source graph')
        report([path])

    if args.only in (None, 'birefnet-lite'):
        print('BiRefNet_lite (1024 px)')
        net = build_birefnet()
        print(f'  loaded strictly, {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params')
        reference = Segmenter(net, 'birefnet').eval()
        images = load_images(args.verify_dir, 1024)
        if args.check_deform:
            with torch.no_grad():
                before = [reference(img) for img in images[:2]]
        from models.modules.deform_conv import DeformableConv2d
        original_forward = DeformableConv2d.forward
        DeformableConv2d.forward = deform_conv_as_grid_sample
        if args.check_deform:
            with torch.no_grad():
                diff = max(float((reference(img) - b).abs().max()) for img, b in zip(images, before))
            print(f'  GridSample rewrite vs torchvision deform_conv2d: max diff {diff:.2e}')
            if diff > 1e-4:
                raise SystemExit('deformable-conv rewrite does not match torchvision')
        export(reference, 1024, tmp)
        DeformableConv2d.forward = original_forward     # reference stays the true upstream op
        m = finalize(onnx.load(tmp))
        check_webgpu_placement(m)

        fp32 = os.path.join(OUT, 'birefnet-lite.fp32.onnx')
        shards = save_sharded(onnx.load_from_string(m.SerializeToString()), fp32, 'birefnet-lite.fp32')
        verify(fp32, reference, images, 1e-3, 'fp32 onnx vs pytorch')

        fp16 = os.path.join(OUT, 'birefnet-lite.fp16.onnx')
        half, _ = dedupe_initializers(to_fp16(m))
        onnx.save_model(half, fp16)
        verify_fp16(fp16, fp32, images)
        report([fp32] + [os.path.join(OUT, s) for s in shards] + [fp16])

    if os.path.exists(tmp):
        os.remove(tmp)
    for leftover in glob.glob(tmp + '*'):
        os.remove(leftover)


if __name__ == '__main__':
    main()
