import mindspore.nn as nn
import numpy as np
import math
import mindspore as ms
import mindspore.ops as ops
from vglaw_final.src.detectron_lib import (
    CNNBlockBase, Conv2d, get_norm, c2_msra_fill, window_partition, window_unpartition, PatchEmbed,
    add_decomposed_rel_pos
)
from functools import partial
from vglaw_final.src.timm_lib import trunc_normal_, DropPath, Mlp
from mindspore.common.initializer import initializer, Constant
from memory_profiler import profile

class Attention(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(in_channels=dim, out_channels=dim * 3, has_bias=qkv_bias)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = ms.Parameter(ms.numpy.zeros((2 * input_size[0] - 1, head_dim)))
            self.rel_pos_w = ms.Parameter(ms.numpy.zeros((2 * input_size[1] - 1, head_dim)))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        # add for evaluation
        # self.last_attn_info = None

    def construct(self, x, ada_weight):
        B, H, W, _ = x.shape
        qkv = self.qkv(x, ada_weight).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.swapaxes(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = ops.softmax(attn, axis=-1)

        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlockBase):
    def __init__(
            self,
            in_channels,
            out_channels,
            bottleneck_channels,
            norm="LN",
            act_layer=nn.GELU,
    ):
        super().__init__(in_channels, out_channels, 1)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, has_bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer(approximate=False)

        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels,
                            3, pad_mode='pad', padding=1)
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer(approximate=False)

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight = layer.weight.fill(1.0)
            layer.bias = layer.bias.fill(0)
        self.norm3.weight = self.norm3.weight.fill(0)
        self.norm3.bias = self.norm3.bias.fill(0)

    def construct(self, x):
        out = x
        for layer in self.cells():
            out = layer(out)

        out = x + out
        return out
    
    
class My_identity(nn.Cell):
    def construct(self, x):
        return x


class Block(nn.Cell):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            window_size=0,
            use_residual_block=False,
            input_size=None,
    ):
        super().__init__()
        self.norm1 = norm_layer([dim, ], epsilon=1e-5)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer([dim, ], epsilon=1e-5)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm='LN',
                act_layer=act_layer,
            )

    def construct(self, x, ada_weight):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x, ada_weight)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x, ada_weight)

        x = shortcut + self.drop_path(x)
        temp_x = self.norm2(x)
        temp_x = self.mlp(temp_x)
        temp_x = self.drop_path(temp_x)
        x = x + temp_x
        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x



class ViT(nn.Cell):

    def __init__(
            self,
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            window_size=0,
            window_block_indexes=(),
            residual_block_indexes=(),
            use_act_checkpoint=False,
            pretrain_img_size=224,
            pretrain_use_cls_token=True,
            out_feature="last_feat",
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patchs = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patchs + 1) if pretrain_use_cls_token else num_patchs
            self.pos_embed = ms.Parameter(ms.Tensor(np.zeros((1, num_positions, embed_dim)), dtype=ms.float32))
        else:
            self.pos_embed = None

        # linspace返回在start和end之间含step个值的一维tensor
        dpr = [x.asnumpy().item() for x in ops.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.CellList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        for m in self.cells():
            if m.cells() == None:
                if isinstance(m, nn.Dense):
                    trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Dense) and m.bias is not None:
                        m.bias = ms.Parameter(ms.numpy.zeros(m.bias.shape),
                                              name=m.bias.name)
                elif isinstance(m, nn.LayerNorm):
                    m.beta = ms.Parameter(ms.numpy.zeros(m.beta.shape), name=m.beta.name)
                    m.gamma = ms.Parameter(ms.numpy.zeros(m.gamma.shape), name=m.gamma.name)
            else:
                for mm in m.cells():
                    if mm.cells() == None:
                        if isinstance(mm, nn.Dense):
                            trunc_normal_(mm.weight, std=0.02)
                            if isinstance(mm, nn.Dense) and mm.bias is not None:
                                mm.bias = ms.Parameter(ms.numpy.zeros(mm.bias.shape),
                                                      name=mm.bias.name)
                        elif isinstance(mm, nn.LayerNorm):
                            mm.beta = ms.Parameter(ms.numpy.zeros(mm.beta.shape), name=mm.beta.name)
                            mm.gamma = ms.Parameter(ms.numpy.zeros(mm.gamma.shape), name=mm.gamma.name)
                    else:
                        for mmm in mm.cells():
                            if mmm.cells() == None:
                                if isinstance(mmm, nn.Dense):
                                    trunc_normal_(mmm.weight, std=0.02)
                                    if isinstance(mmm, nn.Dense) and mmm.bias is not None:
                                        mmm.bias = ms.Parameter(ms.numpy.zeros(mmm.bias.shape),
                                                              name=mmm.bias.name)
                                elif isinstance(mmm, nn.LayerNorm):
                                    mmm.beta = ms.Parameter(ms.numpy.zeros(mmm.beta.shape), name=mmm.beta.name)
                                    mmm.gamma = ms.Parameter(ms.numpy.zeros(mmm.gamma.shape), name=mmm.gamma.name)
                            else:
                                for mmmm in mmm.cells():
                                    if isinstance(mmmm, nn.Dense):
                                        trunc_normal_(mmmm.weight, std=0.02)
                                        if isinstance(mmmm, nn.Dense) and mmmm.bias is not None:
                                            mmmm.bias = ms.Parameter(ms.numpy.zeros(mmmm.bias.shape),
                                                                  name=mmmm.bias.name)
                                    elif isinstance(mmmm, nn.LayerNorm):
                                        mmmm.beta = ms.Parameter(ms.numpy.zeros(mmmm.beta.shape), name=mmmm.beta.name)
                                        mmmm.gamma = ms.Parameter(ms.numpy.zeros(mmmm.gamma.shape), name=mmmm.gamma.name)


    def construct(self, x, ada_weight):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, ada_weight[i])
        outputs = (x.permute(0, 3, 1, 2), )
        return outputs


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = ops.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)

if __name__ == '__main__':
    import pickle

    path = r'/home/zekang/vg-law/mindspore_vglaw/checkpoint/ms_model_final_61ccd1.ckpt'

    # state_dict = torch.load(path)
    ms_dict = ms.load_checkpoint(path)
    # blocks的数量和depth相关
    img_size = 448
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    drop_path_rate = 0.1
    window_size = 14
    mlp_ratio = 4
    qkv_bias = True
    norm_layer = partial(nn.LayerNorm, epsilon=1e-6)
    window_block_indexes = [
        # 2, 5, 8 11 for global attention
        0,
        1,
        3,
        4,
        6,
        7,
        9,
        10,
    ]
    residual_block_indexes = []
    use_rel_pos = True
    out_feature = "last_feat"

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        window_block_indexes=window_block_indexes,
        residual_block_indexes=residual_block_indexes,
        use_rel_pos=use_rel_pos,
        out_feature=out_feature,
    )
    print('LOADING...')
    # pytorch的Attention中_laod_from_state_dict部分修改了预训练模型参数和本模型不同的部分，不然没法load进去
    # 初步验证可行
    for i in range(depth):
        prefix = str(i) + '.' + 'attn' + '.'
        key_rel_pos_h, key_rel_pos_w = prefix + 'rel_pos_h', prefix + 'rel_pos_w'
        if use_rel_pos and key_rel_pos_h in ms_dict:
            rel_pos_h_state = ms_dict[key_rel_pos_h]
            rel_pos_w_state = ms_dict[key_rel_pos_w]
            size_h, size_w = rel_pos_h_state.shape[0], rel_pos_w_state.shape[0]
            # 此处需要self的参数，打算通过model的参数列表获取
            size_h_new = model.parameters_dict()[key_rel_pos_h].shape[0]
            size_w_new = model.parameters_dict()[key_rel_pos_w].shape[0]
            pad_h, pad_w = (size_h - size_h_new)  // 2, (size_w - size_w_new) // 2
            if pad_h > 0:
                ms_dict[key_rel_pos_h] = ms.Parameter(rel_pos_h_state[pad_h: -pad_h])
            if pad_w > 0:
                ms_dict[key_rel_pos_w] = ms.Parameter(rel_pos_w_state[pad_w: -pad_w])
    ms.load_param_into_net(model, ms_dict)
    # model.load_state_dict(state_dict)
    print('LOADING Succeed')
    # model.load_state_dict(checkpoint['state_dict'])
    num = 0
    for n, m in model.cells_and_names():
        if n.endswith('qkv'):
            print(num, n, m)
            num += 1
    print(model.parameters_dict())