from vitdet_vit import ViT
import mindspore.nn as nn
import mindspore as ms
from functools import partial
import os
import pickle


embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
vit_config = dict(
    img_size=448,
    patch_size=16,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    drop_path_rate=dp,
    window_size=14,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    window_block_indexes=[
        # 2, 5, 8 11 for global attention
        0,
        1,
        3,
        4,
        6,
        7,
        9,
        10,
    ],
    residual_block_indexes=[],
    use_rel_pos=True,
    out_feature="last_feat",
)

swin_tiny_config = dict(
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    ape=False,
    drop_path_rate=0.2,
    patch_norm=True,
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    out_indices=(3,),
    use_checkpoint=False,
)

swin_small_config = dict(
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    ape=False,
    drop_path_rate=0.2,
    patch_norm=True,
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    out_indices=(2, 3,),
    # out_indices=(3,),
    use_checkpoint=False,
)

swin_base_config = dict(
    pretrain_img_size=384,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=12,
    ape=False,
    drop_path_rate=0.2,  # 0.3?
    patch_norm=True,
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    out_indices=(2, 3,),
    # out_indices=(3,),
    use_checkpoint=False,
)

NET_CACHE = {
    'vitdet_mae_pretrain_vit_base_norm': 'mae_pretrain_vit_base_norm.pth',
    'vitdet_mae_pretrain_vit_base': 'mae_pretrain_vit_base.pth',
    'swin_tiny_p4_w7_cascade-mrcnn': 'cascade_mask_rcnn_swin_tiny_patch4_window7.pth',
    'swin_tiny_p4_w7_mrcnn': 'ms_mask_rcnn_swin_tiny_patch4_window7.ckpt',
    'vitdet_b_mrcnn': 'ms_model_final_61ccd1.ckpt',
    'vitdet_b_cascade-mrcnn': 'ms_model_final_435fa9.ckpt',
    'vitdet_b_mrcnn_cocoe': 'vitdet_b_mrcnn_cocoe.ckpt',
    'swin_small_mrcnn_unc': 'swin_small_mrcnn_unc_latest.pth',
    'swin_small_mrcnn_gref_umd': 'swin_small_mrcnn_gref_umd_latest.pth',
    'swin_small_mrcnn_no_exclude': 'swin_small_mrcnn_no_exclude.pth',
    'swin_base_patch4_window12_384_22k': 'swin_base_patch4_window12_384_22k.pth',
}


# @ms.jit
def build_backbone(net_num, vit_model_path, **kwargs):
    switch = {1: 'vitdet_mae_pretrain_vit_base_norm',
              2: 'vitdet_mae_pretrain_vit_base',
              3: 'swin_tiny_p4_w7_cascade-mrcnn',
              4: 'swin_tiny_p4_w7_mrcnn',
              5: 'vitdet_b_mrcnn',
              6: 'vitdet_b_cascade-mrcnn',
              7: 'vitdet_b_mrcnn_cocoe',
              8: 'swin_small_mrcnn_unc',
              9: 'swin_small_mrcnn_gref_umd',
              10: 'swin_small_mrcnn_no_exclude',
              11: 'swin_base_patch4_window12_384_22k'}
    if net_num in switch.keys():
        net_name = switch[net_num]
    else:
        raise KeyError(f'Not supported Network number:{net_num}')

    if net_name in NET_CACHE:
        if NET_CACHE[net_name].endswith('.pth'):
            state_dict_total = ms.load(f'/home/zekang/vglaw/checkpoint/{NET_CACHE[net_name]}')
            state_dict = {}
            for k, p in state_dict_total['state_dict'].items():
                if k.startswith('backbone.'):
                    new_k = k[len('backbone.'):]
                    if new_k.startswith(('norm0', 'norm1')):
                        print(f'Skipping {new_k}')
                        continue
                    state_dict[new_k] = p
                    
        elif NET_CACHE[net_name].endswith('.ckpt'):
            ckpt_path = os.path.join(vit_model_path, NET_CACHE[net_name])
            state_dict = ms.load_checkpoint(f'/home/zekang/vglaw/checkpoint/{NET_CACHE[net_name]}')
        else:
            raise KeyError(f'Only support files endswith .ckpt')
    else:
        raise KeyError(f'Not supported Network:{net_name}')

    if net_name.startswith('vitdet'):
        model = ViT(**{**vit_config, **kwargs})
    else:
        raise KeyError(f'Not supported Network:{net_name}')

    unmatched_keys, _ = ms.load_param_into_net(model, state_dict)

    return model
