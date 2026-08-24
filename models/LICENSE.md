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
