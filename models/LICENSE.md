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
