import sys
import math
import mindspore as ms
import mindspore.nn as nn
from bert import BertModel, BertConfig
from mindspore.common.initializer import initializer, HeUniform
import mindspore.ops as ops
from collections import OrderedDict
from cross_attn import CrossAttention
from backbone_vit import build_backbone
from utils import LayerNorm
from xenv import console


class LAWNet(nn.Cell):
    def __init__(self, image_model, vit_model_path, bert_model='bert_base_uncased',
                 text_encoder_layer=6, law_type='svd', groups=None,
                 use_activation=False, max_len=40):
        super(LAWNet, self).__init__()

        self.bert_config = BertConfig(num_hidden_layers=text_encoder_layer)
        self.text_encoder = BertModel(config=self.bert_config)

        print(f"{image_model}\n\n\n")

        switch = {
            'vitdet_mae_pretrain_vit_base_norm': 1,
            'vitdet_mae_pretrain_vit_base': 2,
            'swin_tiny_p4_w7_cascade-mrcnn': 3,
            'swin_tiny_p4_w7_mrcnn': 4,
            'vitdet_b_mrcnn': 5,
            'vitdet_b_cascade-mrcnn': 6,
            'vitdet_b_mrcnn_cocoe': 7,
            'swin_small_mrcnn_unc': 8,
            'swin_small_mrcnn_gref_umd': 9,
            'swin_small_mrcnn_no_exclude': 10,
            'swin_base_patch4_window12_384_22k': 11
        }
        if image_model in switch.keys():
            img_model_num = switch[image_model]
        else:
            raise KeyError(f'Not supported Network:{image_model}')

        self.visual_encoder = build_backbone(img_model_num, vit_model_path)


        if law_type in LAWeight.supported_types:
            self.weight_generator = LAWeight(self.visual_encoder,
                                             self.bert_config.hidden_size,
                                             law_type, groups, use_activation)
        else:
            self.weight_generator = Replace_Lamda()

        kdim = vdim = self.visual_encoder.embed_dim
        qdim = self.bert_config.hidden_size
        self.ada_gap = CrossAttention(qdim, kdim, vdim, adim=64)

        self.up_sample = nn.SequentialCell(
            nn.Conv2dTranspose(self.visual_encoder.embed_dim, 256, kernel_size=2, stride=2, has_bias=True,
                               weight_init='normal', bias_init='normal', pad_mode='valid'),
            LayerNorm(256),
            nn.GELU(approximate=False),
            nn.Conv2dTranspose(256, self.bert_config.hidden_size, kernel_size=2, stride=2, has_bias=True,
                               weight_init='normal', bias_init='normal', pad_mode='valid')
        )
        self.up_sample_final = nn.Upsample(scale_factor=4.0, recompute_scale_factor=True,
                                           mode='bilinear', align_corners=True)
        self.bbox_pred = MLP(self.visual_encoder.embed_dim, 256, 4, 3)


    def construct(self, images, texts):
        text_mask = texts[2].ne(1).bool()
        input_ids = texts[0]
        token_type_ids = texts[1]
        attention_mask = texts[2]
        text_feat = self.text_encoder(input_ids, attention_mask, token_type_ids)
        ada_weight = self.weight_generator(self.visual_encoder, text_feat, text_mask)
        
        img_feat = self.visual_encoder(images, ada_weight)[-1]
        cls_embed = text_feat[:, :1]
        pixel_embed = self.up_sample(img_feat)
        logit_mask = ops.sum(cls_embed.squeeze(1).unsqueeze(-1).unsqueeze(-1) * pixel_embed,
                             dim=1, keepdim=True)
        logit_mask = self.up_sample_final(logit_mask)
        text_feat = text_feat[:, 0].unsqueeze(0)
        img_feat = ops.permute(img_feat.flatten(start_dim=2), (2, 0, 1))
        img_feat = self.ada_gap(text_feat, img_feat, img_feat)[0]
        img_feat = img_feat.squeeze(0)

        pred_box = self.bbox_pred(img_feat)
        pred_box = ops.sigmoid(pred_box)

        return pred_box, logit_mask


class LALinear(nn.Cell):
    def __init__(self, linear_module, **kawrgs):
        super(LALinear, self).__init__(auto_prefix=True)
        assert isinstance(linear_module, nn.Dense)
        self.in_features = linear_module.in_channels
        self.out_features = linear_module.out_channels
        self.weight = ms.numpy.copy(linear_module.weight)
        if linear_module.bias is not None:
            self.bias = ms.numpy.copy(linear_module.bias)
        else:
            self.insert_param_to_cell('bias', None)

        self.ada_weight = None

    def construct(self, input):
        if self.ada_weight is not None:
            C, B = ms.numpy.size(self.weight, axis=1), ms.numpy.size(self.ada_weight, axis=0)
            self.ada_weight = self.ada_weight.view(B, -1, C)
            G = ms.numpy.size(self.ada_weight, axis=1)
            *shape_orig, _ = input.shape
            input = input.view(B, -1, C)
            weight_groups = ops.chunk(self.weight, G)
            bias_group = ops.chunk(self.bias, G) if self.bias is not None else [0]*G
            outputs = []
            for i, (weight_i, bias_i) in enumerate(zip(weight_groups, bias_group)):
                output = ops.einsum('ij,bj,bnj->bni', weight_i, self.ada_weight[:, i], input) + bias_i
                outputs.append(output)
            output = ops.cat(outputs, -1).view(*shape_orig, -1)
            return output
        inc = ms.numpy.size(input, axis=0)
        outc = ms.numpy.size(input, axis=1)
        net = nn.Dense(inc, outc, weight_init=self.weight, bias_init=self.bias)
        return net(input)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ODLinear(nn.Cell):
    def __init__(self, linear_module, **kwargs):
        super(ODLinear, self).__init__(auto_prefix=True)
        assert isinstance(linear_module, nn.Dense)
        self.in_features = linear_module.in_channels
        self.out_features = linear_module.out_channels
        self.weight = ms.numpy.copy(linear_module.weight)
        if linear_module.bias is not None:
            self.bias = ms.numpy.copy(linear_module.bias)
        else:
            self.insert_param_to_cell('bias', None)

        self.ada_weight = None

    def construct(self, input):
        if self.ada_weight is not None:
            (Co, Ci), B = ops.shape(self.weight), ms.numpy.size(self.ada_weight, axis=0)
            n_group = (ms.numpy.size(self.ada_weight, axis=-1) - Co) // Ci
            *shape_orig, _ = input.shape
            input = input.view(B, -1, Ci)
            if n_group == 1:
                BCi, BCo = ops.split(self.ada_weight, (Ci, Co), 1)
                BCi, BCo = ops.unsqueeze(BCi, 1), ops.unsqueeze(BCo, 1)
                input = input * BCi
                inc = ms.numpy.size(input, axis=0)
                outc = ms.numpy.size(input, axis=1)
                net1 = nn.Dense(inc, outc, weight_init=self.weight, bias_init=self.bias)
                output = net1(input*BCi)
            elif n_group == 2:
                BCi_qk, BCi_v, BCo = ops.split(self.ada_weight, (Ci, Ci, Co), 1)
                BCi_qk, BCi_v, BCo = ops.unsqueeze(BCi_qk, 1), ops.unsqueeze(BCi_v, 1), ops.unsqueeze(BCo, 1)
                weight_qk, weight_v = ops.split(self.weight, (Co // 3 * 2, Co // 3))
                bias_qk, bias_v = ops.split(self.bias, (Co // 3 * 2, Co // 3))
                input_qk = input * BCi_qk
                input_v = input * BCi_v
                inc_qk = ms.numpy.size(input_qk, axis=0)
                outc_qk = ms.numpy.size(input_qk, axis=1)
                inc_v = ms.numpy.size(input_v, axis=0)
                outc_v = ms.numpy.size(input_v, axis=1)
                net_qk = nn.Dense(inc_qk, outc_qk, weight_init=weight_qk, bias_init=bias_qk)
                net_v = nn.Dense(inc_v, outc_v, weight_init=weight_v, bias_init=bias_v)
                output_qk = net_qk(input * BCi_qk)
                output_v = net_v(input * BCi_v)
                output = ops.cat([output_qk, output_v], axis=2)
            elif n_group == 3:
                BCi_q, BCi_k, BCi_v, BCo = ops.split(self.ada_weight, (Ci, Ci, Ci, Co), 1)
                BCi_q, BCi_k, BCi_v, BCo = ops.unsqueeze(BCi_q, 1), ops.unsqueeze(BCi_k, 1), ops.unsqueeze(BCi_v,1), ops.unsqueeze(BCo, 1)
                weight_q, weight_k, weight_v = ops.split(self.weight, Co // 3)
                bias_q, bias_k, bias_v = ops.split(self.bias, Co // 3)
                input_q = input * BCi_q
                input_k = input * BCi_k
                input_v = input * BCi_v
                inc_q = ms.numpy.size(input_q, axis=0)
                outc_q = ms.numpy.size(input_q, axis=1)
                inc_k = ms.numpy.size(input_k, axis=0)
                outc_k = ms.numpy.size(input_k, axis=1)
                inc_v = ms.numpy.size(input_v, axis=0)
                outc_v = ms.numpy.size(input_v, axis=1)
                net_q = nn.Dense(inc_q, outc_q, weight_init=weight_q, bias_init=bias_q)
                net_k = nn.Dense(inc_k, outc_k, weight_init=weight_k, bias_init=bias_k)
                net_v = nn.Dense(inc_v, outc_v, weight_init=weight_v, bias_init=bias_v)
                output_q = net_q(input * BCi_q)
                output_k = net_k(input * BCi_k)
                output_v = net_v(input * BCi_v)
                output = ops.cat([output_q, output_k, output_v], axis=2)
            else:
                raise Warning('Too many groups!')

            output = (output * BCo).view(*shape_orig, -1)
            return output
        inc = ms.numpy.size(input, axis=0)
        outc = ms.numpy.size(input, axis=1)
        net = nn.Dense(inc, outc, weight_init=self.weight, bias_init=self.bias)
        return net(input)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SVDLinear(nn.Cell):
    def __init__(self, linear_module, *, dim=32, **kwargs):
        super().__init__()
        assert isinstance(linear_module, nn.Dense)
        self.in_features = linear_module.in_channels
        self.out_features = linear_module.out_channels
        self.weight = ms.Parameter(ms.numpy.copy(linear_module.weight))
        if linear_module.bias is not None:
            self.bias = ms.Parameter(ms.numpy.copy(linear_module.bias))
        else:
            self.insert_param_to_cell('bias', None)

        self.S_D = None

    def construct(self, input, ada_weight):
        '''
        print('进入svd_Linear')
        # self.ada_weight = ada_weight
        print(f"ada_weight:{ada_weight}")
        print(f"S_D:{self.S_D}")
        '''
        if ada_weight is not None:
            Co, Ci, B = ms.numpy.size(self.weight, axis=0), ms.numpy.size(self.weight, axis=1), ms.numpy.size(ada_weight, axis=0)
            K = int(ms.numpy.size(ada_weight, axis=1) ** 0.5)
            ada_weight = ops.reshape(ada_weight, (-1, K, K))
            S, D = self.S_D
            if ms.numpy.size(D, axis=1) > Ci:
                D = D[:, :Ci]
            if ms.numpy.size(S, axis=0) > Co:
                S = (S.reshape(3, -1, ms.numpy.size(S, axis=1))[:, :Co//3]).reshape(-1, ms.numpy.size(S, axis=1), 1)
            shape_orig = list(input.shape[0:3])

            input = input.view(B, -1, Ci)

            out_0 = input.swapaxes(1, 2)
            out_0 = ops.conv1d(out_0, D)
            out_0 = ops.bmm(ada_weight, out_0)
            out_0 = ops.conv1d(out_0, S)
            out_0 = out_0.swapaxes(1, 2)
            out_0 += ops.dense(input, self.weight, self.bias)
            # print(out_0)
            output = out_0.view(*shape_orig, -1)
            return output

        return ops.dense(input, self.weight, self.bias)


class LAWeight(nn.Cell):
    def __init__(self, vit, input_dim, law_type='svd', groups=None, use_activation=False, inner_dim=32, inner_k=32):
        super().__init__()
        self.weights = OrderedDict()
        self.channels = OrderedDict()  # module:(channels_in, channels_out)
        self.groups = OrderedDict()  # module: group number
        self.qkvs = OrderedDict()

        ################### add for evaluation ###################
        self.attn_layers = OrderedDict()
        ##########################################################

        index = 0
        for _, m in vit.cells_and_names():
            if hasattr(m, 'qkv'):
                ################### add for evaluation ###################
                self.attn_layers[f'attn_{index}'] = m
                ##########################################################
                orig_linear = getattr(m, 'qkv')
                setattr(m, 'qkv', self.supported_types[law_type](orig_linear, dim=inner_k))
                linear = getattr(m, 'qkv')
                co, ci = linear.weight.shape
                module_name = f'qkv_{index}'
                self.channels[module_name] = (ci, co)
                self.qkvs[module_name] = linear
                if groups is None or groups[index] > 0:
                    self.groups[module_name] = 1 if groups is None else groups[index]
                index += 1

        max_ci, max_co = 0, 0
        for n in self.groups:  # n = qkv_n
            ci, co = self.channels[n]
            max_ci, max_co = max(max_ci, ci), max(max_co, co)
        if 'svd' in law_type:
            ######### init #######
            S = initializer(HeUniform(negative_slope=math.sqrt(5)), [max_co, inner_k, 1], ms.float32)
            D = initializer(HeUniform(negative_slope=math.sqrt(5)), [inner_k, max_ci, 1], ms.float32)
            ######################
            self.S = ms.Parameter(S)
            self.D = ms.Parameter(D)
            for n in self.groups:
                self.qkvs[n].S_D = (self.S, self.D)

        self.num_layer = num_layer = len(self.groups)

        ####################################################
        inner_dim = input_dim // 16
        ####################################################

        layer_embedding = initializer(HeUniform(negative_slope=math.sqrt(5)), [num_layer, input_dim], ms.float32)
        self.layer_embedding = ms.Parameter(layer_embedding.unsqueeze(1))  # L1C
        self.attn = CrossAttention(input_dim, input_dim, input_dim, num_heads=12)

        self.proj = nn.CellList()
        for m in self.groups.keys():
            ci, co = self.channels[m]
            if law_type.startswith('la'):
                n_channel = self.groups[m] * ci
            elif law_type.startswith('od'):
                n_channel = co + ci * int(law_type[2])
            elif law_type.startswith('svd'):
                n_channel = inner_k ** 2
            else:
                raise ValueError(f'unsupported law_type-{law_type}, only support law_type starts with la/od/svd')
            proj = nn.SequentialCell(
                nn.Dense(input_dim, inner_dim),
                nn.LayerNorm((inner_dim, )),
                nn.GELU(approximate=False),
                nn.Dense(inner_dim, n_channel)
            )
            ## add constant initialization if no activation func used! ##
            if use_activation:
                proj[-1].weight = ms.Parameter(ms.numpy.zeros(proj[-1].weight.shape, ms.float32))
                proj[-1].bias = ms.Parameter(ms.numpy.ones(proj[-1].bias.shape, ms.float32))
            else:
                proj[-1].weight = ms.Parameter(ms.numpy.zeros(proj[-1].weight.shape, ms.float32))
                proj[-1].bias = ms.Parameter(ms.numpy.ones(proj[-1].bias.shape, ms.float32))
            if law_type.startswith('svd'):
                proj[-1].weight = ms.Parameter(ms.numpy.zeros(proj[-1].weight.shape, ms.float32))
                proj[-1].bias = ms.Parameter(ms.numpy.zeros(proj[-1].bias.shape, ms.float32))
                print('zero the weight and bias for SVD')
            self.proj.append(proj)

        self.use_activation = use_activation
        console.print(f'{num_layer} projection layer hooked! ## SVDConv, Multi-task Upsample ##')

        ############## extra added for evaluation #############
        self.without_ada_weight = False

    # subscript object has no attribute id
    def construct(self, vit, text_feats, text_masks):


        ada_weight = []

        if self.without_ada_weight:
            for i in range(len(self.proj)):
                m = 'qkv_' + str(i)
            return None

        B = text_masks.shape[0]
        a = self.layer_embedding.shape[0]
        c = self.layer_embedding.shape[2]
        temp_tensor = ms.numpy.zeros((a, B, c))
        text_feats = text_feats.swapaxes(0, 1)  # BLC->LBC
        layer_embedding = self.layer_embedding.expand_as(temp_tensor)
        inner_feat = self.attn(layer_embedding,
                               text_feats,
                               text_feats,
                               text_masks)[0]  # NBC
        for i in range(len(self.proj)):
            m = 'qkv_' + str(i)
            feat = inner_feat[i]  # BC
            scales = self.proj[i](feat)
            if self.use_activation:
                scales = ops.relu(scales)
            ada_weight.append(scales)
        return ada_weight

        ################### add for evaluation ###################
    # @ms.jit
    def get_attn_info(self):
        attn_info = []
        for n, m in self.attn_layers.items():
            attn_info.append((n, m.last_attn_info))
        return attn_info

    supported_types = {    
        'la': LALinear,
        'od1': ODLinear,
        'od2': ODLinear,
        'od3': ODLinear,
        'svd': SVDLinear,
    }


class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__(auto_prefix=True)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList()
        for n, k in zip([input_dim] + h, h + [output_dim]):
            self.layers.append(ms.nn.Dense(n, k))

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.relu(x)
        return x
    

class Replace_Lamda(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, y, z):
        return None, None

 
def get_network(args, num_hidden_layers):
    image_model = args.vit_model
    groups = args.groups
    use_activation = args.use_activation
    net = LAWNet(image_model, args.vit_model_path, text_encoder_layer=num_hidden_layers,
                    law_type='svd', groups=groups,
                    use_activation=use_activation,)
    for name, data in net.parameters_and_names():
        if name.endswith('qkv.weight'):
            data.name = name
        elif name.endswith('qkv.bias'):
            data.name = name
    return net
