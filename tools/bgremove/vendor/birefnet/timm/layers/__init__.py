import collections.abc
from itertools import repeat

import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


class DropPath(nn.Module):
    """Stochastic depth. Identity at inference, which is all an export needs."""
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Initialisation only; the checkpoint overwrites every value.
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean, std, a, b)
