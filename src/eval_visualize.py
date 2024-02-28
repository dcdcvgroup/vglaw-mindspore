import time
import numpy as np
import vglaw_final.src.xenv as xenv
import mindspore as ms
import mindspore.dataset.vision as V
from mindspore.dataset.vision import Inter
import os
from vglaw_final.src.dataset import ReferDataset, create_dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model_law import LAWNet
import time
ms.set_seed(0)

eval_set_index = 0 # varies by dataset, such as refcoco has val/testA/testB sets, it can be 0/1/2
index = 1300  # the index of target image
pretrained_path = '/home/zekang/vglaw/checkpoint/law_fpt_vit_refcoco_shareSD_mt.ckpt'
output_path = '/home/zekang/vglaw/visualize/eval_visualize.png'
cfgs = '''
--batch_size=1
--dataset=refcoco
--splitBy=unc
--experiment_name=visualize
--short_comment=visualize
--use_mask
--law_type=svd
--img_size=448
--vit_model=vitdet_b_mrcnn
'''.strip().replace('\n', '=').split('=')
args = xenv.parse_args(cfgs)


mode = 0
if mode == 0:
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
elif mode == 1:
    ms.context.set_context(mode=ms.context.GRAPH_MODE)

t1 = time.time()
os.environ['CUDA_VISIBEL_DEVICE'] = '0'
ms.set_context(device_target='GPU', device_id=0)


eval_dataset = create_dataset(args, batch_size=args.batch_size, split=args.train_split, is_train=False,
                                   rank=0, group_size=1, shuffle=True)[0].create_tuple_iterator()

count = 0
print('locating the data......')
for i in eval_dataset:
    if count == index:
        images, target_boxes, texts, raw_text, ref_masks, masks = i
        break
    else:
        count += 1

model = LAWNet(args.vit_model, vit_model_path=args.vit_model_path, 
               bert_model='bert_base_uncased',
               text_encoder_layer=6, law_type=args.law_type,
               groups=args.groups, use_activation=args.use_activation,
               max_len=args.max_len)


for name, data in model.parameters_and_names():
    if name.endswith('qkv.weight'):
        data.name = name
    elif name.endswith('qkv.bias'):
        data.name = name

model.set_train(False)
for m in model.get_parameters():
    m.requires_grad = False

param_dict = ms.load_checkpoint(pretrained_path)
param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
print(f'param_not_load:{param_not_load}')
print(f'ckpt_not_load:{ckpt_not_load}')

######### switch #########
# model.weight_generator.without_ada_weight = True
model.weight_generator.without_ada_weight = False
##########################

pred_box, logit_mask = model(images, texts)
print(f'pred_box:{pred_box}')
print(f'logit_mask:{logit_mask}')
pred_mask = logit_mask.sigmoid() > args.mask_threshold

# pred_mask = logit_mask.sigmoid() > 0.1

mean = ms.Tensor((0.485, 0.456, 0.406))
std = ms.Tensor((0.229, 0.224, 0.225))

image = (ms.ops.permute(images[0], (1, 2, 0)) * std + mean).asnumpy()
print(f"shape of image:{image.shape}")
print(raw_text)

plt.figure(dpi=180)

plt.subplot(141)
plt.axis('off')
plt.imshow(image)
plt.title('box_gt')
print(target_boxes[0])
gt_cx, gt_cy, gt_w, gt_h = (target_boxes[0].asnumpy() * args.img_size).tolist()
rect_gt = Rectangle((gt_cx-0.5*gt_w, gt_cy-0.5*gt_h), gt_w, gt_h, fill=False, color='g', linewidth=2)
plt.gca().add_patch(rect_gt)

plt.subplot(142)
plt.title('box_pred')
plt.axis('off')
plt.imshow(image)
gt_cx, gt_cy, gt_w, gt_h = (pred_box[0].asnumpy() * args.img_size).tolist()
print(pred_box[0])
print(gt_cx, gt_cy, gt_w, gt_h)
rect_gt = Rectangle((gt_cx-0.5*gt_w, gt_cy-0.5*gt_h), gt_w, gt_h, fill=False, color='b', linewidth=2)
plt.gca().add_patch(rect_gt)


alpha = 0.5
mask_color = ms.Tensor((0.1, 0.8, 0.1))
plt.subplot(143)
plt.title('mask_gt')
plt.axis('off')
img_mask = image + (ref_masks[0].squeeze(0).unsqueeze(-1) * mask_color).asnumpy()
plt.imshow(img_mask)

plt.subplot(144)
plt.title('mask_pred')
plt.axis('off')

pred_mask = pred_mask.numpy()
pred_mask = pred_mask.astype(float)
pred_mask = np.transpose(pred_mask, [0, 2, 3, 1])
tmp_mask = V.Resize(args.img_size, interpolation=Inter.LINEAR)(pred_mask)
tmp_mask = np.transpose(tmp_mask, [0, 3, 1, 2])
tmp_mask = tmp_mask > 0
tmp_mask = tmp_mask.astype(bool)
tmp_mask = tmp_mask[0, 0]


tmp_mask = np.expand_dims(tmp_mask, axis=-1)
tmp_mask = tmp_mask * mask_color.asnumpy()
img_mask = image + tmp_mask
plt.imshow(img_mask)
plt.show()


plt.savefig(output_path)

t2 = time.time() - t1
print(f'耗时{t2}')