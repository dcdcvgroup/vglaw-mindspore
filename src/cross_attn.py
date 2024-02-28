import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np


class CrossAttention(nn.Cell):
    def __init__(self, qdim, kdim, vdim, odim=None, adim=None, num_heads=1, bias=True, proj=False):
        super(CrossAttention, self).__init__(auto_prefix=True)
        self.num_heads = num_heads
        if proj or (qdim != kdim) or (adim is not None):
            self.q_proj = nn.Dense(qdim, adim, bias)
            self.k_proj = nn.Dense(kdim, adim, bias)
        else:
            self.q_proj = self.insert_child_to_cell('q_proj', None)
            self.k_proj = self.insert_child_to_cell('k_proj', None)
        if odim is not None:
            self.o_proj = nn.Dense(vdim, odim, bias)
        else:
            self.o_proj = self.insert_child_to_cell('o_proj', None)

    def construct(self, query, key, value, key_padding_mask=None):
        q, k, v = query, key, value
        if self.q_proj is not None:
            q = self.q_proj(q)
            k = self.k_proj(k)

        tgt_len, B, adim = np.size(q, axis=0), np.size(q, axis=1), np.size(q, axis=2)
        src_len, vdim = np.size(k, axis=0), np.size(v, axis=-1)
        num_heads = self.num_heads
        q = ops.swapaxes(ops.reshape(q, (tgt_len, B * num_heads, adim // num_heads)), 0, 1)
        k = ops.swapaxes(ops.reshape(k, (src_len, B * num_heads, adim // num_heads)), 0, 1)
        v = ops.swapaxes(ops.reshape(v, (src_len, B * num_heads, vdim // num_heads)), 0, 1)

        attn_mask = None
        if key_padding_mask is not None:
            temp_tensor = ms.numpy.zeros((B, num_heads, src_len))
            key_padding_mask = key_padding_mask.view(B, 1, src_len)
            key_padding_mask = key_padding_mask.expand_as(temp_tensor)
            key_padding_mask = key_padding_mask.reshape(B * num_heads, 1, src_len)
            attn_mask = key_padding_mask
            if key_padding_mask.dtype == ms.bool_:
                attn_mask = ops.zeros_like(key_padding_mask, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        E = np.size(q, axis=-1)
        q = q.div(math.sqrt(E))
        attn = ops.bmm(q, k.swapaxes(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        attn = ops.softmax(attn, axis=-1)
        attn_output = ops.bmm(attn, v)
        attn_output = ops.swapaxes(attn_output, 0, 1).reshape(tgt_len, B, vdim)
        if self.o_proj is not None:
            attn_output = self.o_proj(attn_output)

        attn = attn.view(B, num_heads, tgt_len, src_len)

        return attn_output, attn


if __name__ == '__main__':
    input_dim, num_heads = 768, 12
    B = 8
    cattn = CrossAttention(input_dim, input_dim, input_dim, num_heads=num_heads)
    q = ms.numpy.randn(12, B, input_dim)
    k = v = ms.numpy.randn(15, B, input_dim)
    mask = ms.numpy.randn(B, 15).ge(0)
    out1, out2 = cattn(query=q, key=k, value=v, key_padding_mask=mask)

