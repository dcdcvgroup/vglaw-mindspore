import math
import mindspore as ms
import warnings
import mindspore.nn as nn
import mindspore.ops as ops


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor = ops.uniform(tensor.shape,
                         ms.Tensor(2 * l - 1, ms.float32), ms.Tensor(2 * u - 1, ms.float32))
    tensor = tensor.erfinv()
    tensor = tensor.mul(std * math.sqrt(2.))
    tensor = tensor.add(mean)
    tensor = tensor.clamp(min=a, max=b)
    l = ops.stop_gradient(l)
    u = ops.stop_gradient(u)
    tensor = ops.stop_gradient(tensor)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ops.rand(shape, dtype=x.dtype)
    random_tensor = ops.floor(random_tensor)
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Cell):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer(approximate=False)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
