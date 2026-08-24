"""Export Real-ESRGAN family checkpoints to fully-dynamic ONNX graphs.

Architectures are re-implemented here (rather than pulling in basicsr) so the
export has no heavyweight dependencies and the state dicts are loaded strictly.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return torch.clamp(out, 0.0, 1.0)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return torch.clamp(self.conv_last(self.lrelu(self.conv_hr(feat))), 0.0, 1.0)


def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('params_ema', 'params'):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt


def blend(state_a, state_b, weight_a):
    """Deep network interpolation, matching Real-ESRGAN's --denoise_strength."""
    return {k: state_a[k] * weight_a + state_b[k] * (1 - weight_a) for k in state_a}


WEIGHTS = 'weights'


def build(spec):
    kind = spec['arch']
    if kind == 'compact':
        model = SRVGGNetCompact(num_conv=spec['num_conv'], upscale=spec['scale'])
    else:
        model = RRDBNet(scale=spec['scale'], num_block=spec['num_block'])
    if 'blend' in spec:
        a, b, w = spec['blend']
        state = blend(load_state(os.path.join(WEIGHTS, a)), load_state(os.path.join(WEIGHTS, b)), w)
    else:
        state = load_state(os.path.join(WEIGHTS, spec['ckpt']))
    model.load_state_dict(state, strict=True)
    return model.eval()


SPECS = {
    'realesr-general-x4v3-balanced': dict(
        arch='compact', num_conv=32, scale=4,
        blend=('realesr-general-wdn-x4v3.pth', 'realesr-general-x4v3.pth', 0.5)),
    'realesr-general-x4v3-detail': dict(
        arch='compact', num_conv=32, scale=4, ckpt='realesr-general-x4v3.pth'),
    'realesr-general-x4v3-denoise': dict(
        arch='compact', num_conv=32, scale=4, ckpt='realesr-general-wdn-x4v3.pth'),
    'realesr-animevideov3-x4': dict(
        arch='compact', num_conv=16, scale=4, ckpt='realesr-animevideov3.pth'),
    'realesrgan-x4plus-anime-6b': dict(
        arch='rrdb', num_block=6, scale=4, ckpt='RealESRGAN_x4plus_anime_6B.pth'),
    'realesrgan-x4plus': dict(
        arch='rrdb', num_block=23, scale=4, ckpt='RealESRGAN_x4plus.pth'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='onnx')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, spec in SPECS.items():
        if args.only and name not in args.only:
            continue
        print(f'--- {name}', flush=True)
        model = build(spec)
        dummy = torch.rand(1, 3, 64, 64)
        dest = os.path.join(args.out, name + '.onnx')
        torch.onnx.export(
            model, dummy, dest,
            export_params=True, opset_version=args.opset, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            # The output dims MUST NOT reuse the input's dim_param names. ONNX
            # treats a repeated dim_param as an assertion that the two sizes are
            # equal, so naming both 'height' tells the runtime that a 4x-larger
            # output is the same size as its input. ONNX Runtime's allocation
            # planner then aliases the output onto the input buffer and fails at
            # run time with "Shape mismatch attempting to re-use buffer".
            dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                          'output': {0: 'batch', 2: 'height_out', 3: 'width_out'}},
            dynamo=False)
        size = os.path.getsize(dest) / 1e6
        print(f'    wrote {dest} ({size:.1f} MB)', flush=True)


if __name__ == '__main__':
    sys.exit(main())
