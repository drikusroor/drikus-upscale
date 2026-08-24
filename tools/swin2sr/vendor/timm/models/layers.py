"""
Minimal stand-ins for the three timm.models.layers helpers network_swin2sr.py
imports, so the upstream architecture file can be used completely unmodified
without pulling in the full timm package. All three only matter for training
(weight init, stochastic depth) -- inference with a loaded checkpoint in
eval() mode is unaffected by using these instead of the real implementations.
"""
import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Only used inside Swin2SR.__init__ before the real checkpoint is loaded
    # over every weight, so an exact truncated-normal is not required here.
    with torch.no_grad():
        return tensor.normal_(mean, std)


class DropPath(nn.Module):
    """Stochastic depth -- a no-op in eval mode, which is all we ever run."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x
