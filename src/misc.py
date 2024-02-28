import collections.abc
from itertools import repeat

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
import mindspore as ms


def _ntuple(n, x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, n))


class Identity(nn.Cell):
    """Identity"""
    def construct(self, x):
        return x


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath1D(DropPath):
    def __init__(self, drop_prob):
        super(DropPath1D, self).__init__(drop_prob=drop_prob, ndim=1)
