"""
Export the Swin2SR compressed-SR checkpoint to a fixed-shape ONNX graph.

Unlike tools/export_onnx.py (the RRDBNet/SRVGGNetCompact export, which produces
fully dynamic-shape graphs), this one is deliberately static-shape. Swin2SR's
shifted-window attention blocks build an attention mask from the *current*
input resolution (see WindowAttention / calculate_mask in vendor/network_swin2sr.py);
when traced for ONNX export at one concrete size, that mask is baked into the
graph as a constant correct for that size only. Feeding any other resolution at
inference would silently produce the wrong mask rather than error, so instead
this script traces at exactly the tile shape assets/models.js declares
(fixedTile + 2*fixedContextPad), and the app forces every call to that model to
use that exact shape (see worker.js's fixedModel handling).

Usage:
    mkdir -p weights && cd weights
    curl -LO https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth
    cd ..
    pip install torch onnx onnxruntime
    python tools/swin2sr/export.py

vendor/network_swin2sr.py is the upstream architecture file, used unmodified
(Apache-2.0, see vendor/LICENSE) from https://github.com/mv-lab/swin2sr. It
imports three small utilities from `timm.models.layers`; vendor/timm/ is a
minimal stand-in for just those three (see its module docstring) so the
architecture file can be used as-is without installing timm.
"""
import os
import sys

# Both the architecture file and the timm shim live directly under vendor/,
# so putting just that one directory on the path resolves both
# `from network_swin2sr import ...` and network_swin2sr.py's own `import timm`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor'))

import torch
import torch.nn as nn

from network_swin2sr import Swin2SR as Net

TILE = 128   # fixedTile (96) + 2 * fixedContextPad (16) in assets/models.js -- must match exactly
WEIGHTS = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'Swin2SR_CompressedSR_X4_48.pth')
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'swin2sr-compressed-x4.onnx')


class SingleOutput(nn.Module):
    """Swin2SR's pixelshuffle_aux head returns (sr, aux); the app only wants sr."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        sr, _aux = self.model(x)
        return sr


def main():
    # Args match define_model()'s 'compressed_sr' branch in the upstream
    # main_test_swin2sr.py exactly -- these are architecture hyperparameters,
    # not tuning knobs, and must match what the checkpoint was trained with.
    model = Net(
        upscale=4, in_chans=3, img_size=48, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler='pixelshuffle_aux', resi_connection='1conv',
    )
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    state = state['params'] if isinstance(state, dict) and 'params' in state else state
    model.load_state_dict(state, strict=True)
    model.eval()
    print('loaded, params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')

    wrapped = SingleOutput(model).eval()
    dummy = torch.rand(1, 3, TILE, TILE)
    with torch.no_grad():
        ref = wrapped(dummy)
    print('reference output shape:', ref.shape)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, OUT,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes=None,   # fixed shape by design -- see module docstring
        dynamo=False,
    )
    size = os.path.getsize(OUT) / 1e6
    print(f'wrote {OUT} ({size:.1f} MB)')


if __name__ == '__main__':
    main()
