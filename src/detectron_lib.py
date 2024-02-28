import mindspore as ms
import mindspore.nn as nn
import warnings
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeNormal, Constant
import numpy as np


class FrozenBatchNorm2d(nn.Cell):
    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = ms.Parameter(ms.numpy.ones(num_features), requires_grad=False)
        self.bias = ms.Parameter(ms.numpy.zeros(num_features), requires_grad=False)
        self.moving_mean = ms.Parameter(ms.numpy.zeros(num_features), requires_grad=False)
        self.moving_var = ms.Parameter(ms.numpy.ones(num_features) - eps, requires_grad=False)

    # @ms.jit
    def construct(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return ops.batch_norm(x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )


    def converat_frozen_batchnorm(cls, module):
        bn_module = (nn.BatchNorm2d, nn.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = ms.Parameter(module.weight.data.clone().asnumpy())
                res.bias.data = ms.Parameter(module.bias.data.clone().asnumpy())
            res.moving_mean = module.moving_mean
            res.moving_variance = module.moving_variance
            res.eps = module.eps
        else:
            i = 0
            for name, cell in module.cells_and_names()():
                if i == 0:
                    i += 1
                    continue
                new_cell = cls.converat_frozen_batchnorm(cell)
                if new_cell is not cell:
                    res.insert_child_to_cell(name, new_cell)

        return res

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

class CNNBlockBase(nn.Cell):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.get_parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class Conv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def construct(self, x):
        x = ops.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.group)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class LayerNorm(nn.Cell):
    def __init__(self, normlized_shape, eps=1e-6):
        super().__init__()
        self.weight = ms.Parameter(ms.numpy.ones(normlized_shape))
        self.bias = ms.Parameter(ms.numpy.zeros(normlized_shape))
        self.eps = eps
        self.normalized_shape = (normlized_shape, )

    def construct(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / ms.numpy.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm, 
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def c2_msra_fill(Cell: nn.Cell) -> None:
    Cell.weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), Cell.weight.shape)
    if Cell.bias is not None:
        Cell.bias = initializer(Constant(0), Cell.bias.shape)


def window_partition(x, window_size):
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = ops.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Cell):
    def __init__(
            self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode='pad',
                              has_bias=True)

    def construct(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    Rh_temp = Rh.permute(0, 2, 1)
    rel_h = ops.matmul(r_q, Rh_temp)
    Rw_temp = Rw.permute(0, 2, 1)
    r_q_temp = r_q.unsqueeze(3)
    rel_w = ops.matmul(r_q_temp, Rw_temp).squeeze(3)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = ops.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                                          size=max_rel_dist, mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = ms.numpy.arange(q_size)[:, None] * ms.Tensor(max(k_size / q_size, 1.0))
    k_coords = ms.numpy.arange(k_size)[None, :] * ms.Tensor(max(q_size / k_size, 1.0))
    relative_coords = (q_coords - k_coords) + ms.Tensor(k_size - 1) * ms.Tensor(max(q_size / k_size, 1.0))
    return rel_pos_resized[relative_coords.to(ms.int64)]
