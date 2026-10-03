# Model weights

The `.onnx` files in this directory are ONNX conversions of the pretrained
Real-ESRGAN checkpoints published at
<https://github.com/xinntao/Real-ESRGAN/releases>.

Real-ESRGAN is Copyright (c) 2021, Xintao Wang, and is released under the
BSD 3-Clause License. See
<https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE> for the full text.

`realesr-general-x4v3-balanced.onnx` is a 50/50 deep network interpolation of
`realesr-general-wdn-x4v3` and `realesr-general-x4v3`, equivalent to running the
upstream inference script with `--denoise_strength 0.5`.

`tools/export_onnx.py` in this repository reproduces every file here from those
checkpoints.

`swin2sr-compressed-x4.onnx` is an ONNX export of the `Swin2SR_CompressedSR_X4_48`
checkpoint from [Swin2SR](https://github.com/mv-lab/swin2sr) (Conde, Choi, Burchi
and Timofte, 2022), published at
<https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1> under the Apache License
2.0 (see `tools/swin2sr/vendor/LICENSE`). `tools/swin2sr/export.py` reproduces it
from that checkpoint.

## Background removal

`birefnet-lite.fp32.onnx` (with its external-data shards
`birefnet-lite.fp32.data0` and `.data1`) and `birefnet-lite.fp16.onnx` are ONNX
exports of the `BiRefNet-general-bb_swin_v1_tiny-epoch_232` checkpoint
(BiRefNet_lite) from [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
(Zheng, Gao, Fan, Liu, Laaksonen, Ouyang and Sebe, 2024), published at
<https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1>. BiRefNet is
Copyright (c) 2024 ZhengPeng and released under the MIT License (see
`tools/bgremove/vendor/birefnet/LICENSE`).

`u2netp.onnx` is an ONNX export of the U²-Netp checkpoint from
[U-2-Net](https://github.com/xuebinqin/U-2-Net) (Qin, Zhang, Huang, Dehghan,
Zaiane and Jagersand, 2020), released under the Apache License 2.0 (see
`tools/bgremove/vendor/u2net/LICENSE`). Upstream publishes `u2netp.pth` on
Google Drive only; the weights here were recovered exactly from
[rembg](https://github.com/danielgatis/rembg)'s ONNX conversion of that same
checkpoint, as `tools/bgremove/export.py` documents.

`tools/bgremove/export.py` reproduces all of these files from those
checkpoints.
