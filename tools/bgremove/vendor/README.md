# Vendored architectures

Upstream model code, used unmodified by `tools/bgremove/export.py`:

| Directory | Upstream | Commit | Licence |
|---|---|---|---|
| `birefnet/models/birefnet.py`, `birefnet/models/backbones/swin_v1.py`, `birefnet/models/modules/{aspp,decoder_blocks,deform_conv,lateral_blocks}.py` | <https://github.com/ZhengPeng7/BiRefNet> | `ebcc0bc8ec7fe919cec829f2dea656b3078acddc` | MIT (`birefnet/LICENSE`) |
| `u2net/u2net.py` | <https://github.com/xuebinqin/U-2-Net> (`model/u2net.py`) | `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29` | Apache-2.0 (`u2net/LICENSE`) |

Everything else under `birefnet/` is a small stand-in written for this
repository so those files import without their training-time dependencies:
`config.py` (the BiRefNet_lite settings, without the file-system probing),
`dataset.py`, `models/backbones/build_backbone.py` (Swin only), and minimal
`timm`, `huggingface_hub` and `kornia` shims. Each says what it replaces in
its docstring.
