import math
import mindspore as ms
from mindspore import Tensor, ops, Parameter
import mindspore.nn as nn
from mindspore.ops import functional as F
from utils import box_ious_v2, sigmoid_focal_loss, dice_loss
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean, _get_parallel_mode, _is_pynative_parallel
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter


class loss_bbox_xiou(nn.Cell):
    def __init__(self, reduction, xiou_loss_type, bbox_loss_coef, xiou_loss_coef, focal_loss_coef, dice_loss_coef):
        super(loss_bbox_xiou, self).__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.xiou_loss_type = xiou_loss_type
        self.bbox_loss_coef = bbox_loss_coef
        self.xiou_loss_coef = xiou_loss_coef
        self.focal_loss_coef = focal_loss_coef
        self.dice_loss_coef = dice_loss_coef

    def construct(self, pred_box, logit_mask, bbox, ref_mask):
        loss_bbox = self.l1_loss(pred_box, bbox) / bbox.size(0)
        loss_xiou = (1 - box_ious_v2(pred_box, bbox, self.xiou_loss_type)[self.xiou_loss_type]).mean()
        loss_focal = sigmoid_focal_loss(logit_mask, ref_mask)
        loss_dice = dice_loss(logit_mask, ref_mask)
        total_loss = (
            self.bbox_loss_coef * loss_bbox +
            self.xiou_loss_coef * loss_xiou +
            self.focal_loss_coef * loss_focal +
            self.dice_loss_coef * loss_dice
        )
        return total_loss

class loss_bbox(nn.Cell):
    def __init__(self, red):
        super().__init__()
        self.l1loss = nn.L1Loss(reduction=red)
    def construct(self, logits, label):
        return self.l1loss(logits, label) / label.shape[0]

class box_iou(nn.Cell):
    def construct(self, boxes1, boxes2, types='iou'):
        types = [types] if isinstance(types, str) else types
        results = {}
        boxes2=boxes2.astype(boxes1.dtype)

        b1_cxcy = boxes1[:, :2]  # center x and center y of boxes1, (N,(cx,cy))
        b2_cxcy = boxes2[:, :2]  # center x and center y of boxes2, (N,(cx,cy))
        # half of the width and height of boxes1, (N,(0.5w,0.5h))
        b1_wh_half = 0.5 * boxes1[:, 2:]
        # half of the width and height of boxes2, (N,(0.5w,0.5h))
        b2_wh_half = 0.5 * boxes2[:, 2:]

        b1_lt = b1_cxcy - b1_wh_half  # left top of boxes1, (N,(lt_x,lt_y))
        b1_rb = b1_cxcy + b1_wh_half  # right bottom of boxes1, (N,(rb_x,rb_y))
        b2_lt = b2_cxcy - b2_wh_half  # left top of boxes2, (N,(lt_x,lt_y))
        b2_rb = b2_cxcy + b2_wh_half  # right bottom of boxes2, (N,(rb_x,rb_y))

        # dependence: iou, giou, diou, ciou
        intersect_lt = ops.maximum(b1_lt, b2_lt)  # (N,(x,y))
        intersect_rb = ops.minimum(b1_rb, b2_rb)
        dif = intersect_rb - intersect_lt  # (N,(dif_x,dif_y))
        clip_max_value = 10
        intersect_wh = ops.clip_by_value(dif, clip_value_min=0, clip_value_max=clip_max_value)  # (N,(w,h))
        intersect_area = ops.prod(intersect_wh, axis=1)  # (N,area)
        union_area = ops.prod(boxes1[:, 2:], 1) + ops.prod(boxes2[:, 2:], 1) - intersect_area  # (N,area)
        clip_max_value = 10
        # iou = intersect_area / union_area.clamp(min=1e-6)  # N
        iou = intersect_area / ops.clip_by_value(union_area, clip_value_min=1e-6, clip_value_max=clip_max_value)
        results['iou'] = iou
        # dependence: giou, diou, ciou

        # init parameters to avoid bug in graph mode
        enclose_lt = 0
        enclose_rb = 0
        enclose_wh = 0
        enclose_area = 0
        diou = 0
        if 'giou' in types or 'diou' in types or 'ciou' in types:
            enclose_lt = ops.minimum(b1_lt, b2_lt)
            enclose_rb = ops.maximum(b1_rb, b2_rb)
        # dependence: giou
        if 'giou' in types:
            enclose_wh = (enclose_rb - enclose_lt)  # N,2
            enclose_area = ops.prod(enclose_wh, 1)  # N
            giou = iou - (enclose_area - union_area) / enclose_area
            results['giou'] = giou
        # dependence: diou, ciou
        if 'diou' in types or 'ciou' in types:
            center_dist2 = ops.reduce_sum(ops.pow(boxes1[:, :2] - boxes2[:, :2], 2), axis=1)  # (N)
            enclose_dist2 = ops.reduce_sum(ops.pow(enclose_lt - enclose_rb, 2), axis=1)  # (N)
            diou = iou - center_dist2 / enclose_dist2
            results['diou'] = diou
        # dependence: ciou
        if 'ciou' in types:
            arctan = ops.atan(boxes1[:, 2]/boxes1[:, 3]) - \
                ops.atan(boxes2[:, 2]/boxes2[:, 3])  # N
            v = (4 / math.pi**2) * ops.pow(arctan, 2)
            alpha = v / (1 - iou + v + 1e-6)
            ciou = diou - alpha * v
            results['ciou'] = ciou
        return results
    
class loss_xiou(nn.Cell):
    def __init__(self, xiou_loss_type):
        super().__init__()
        self.box_iou_fn = box_iou()
        self.xiou_loss_type = xiou_loss_type
    def construct(self, logits, labels):
        result = (1 - self.box_iou_fn(logits, labels, self.xiou_loss_type)[self.xiou_loss_type]).mean()
        return result
    
class loss_focal(nn.Cell):
    def __init__(self):
        super().__init__()
        # self.resize_fn = vision.Resize(size=[896, 896], interpolation=Inter.NEAREST)
    
    def construct(self, logits, labels, alpha: float = 0.25, gamma: float = 2):
        # labels = labels.squeeze(1).permute(1, 2, 0)
        # labels = self.resize_fn(labels.numpy()).transpose(2, 0, 1)
        # labels = ms.Tensor(labels, ms.float32)
        # logits = logits.squeeze(1)
        
        labels = labels.flatten(start_dim=0)
        logits = logits.flatten(start_dim=0)
        preds = ops.sigmoid(logits)
        ce_loss = ops.binary_cross_entropy_with_logits(logits, labels, weight=ms.Tensor([1.0], ms.float32),
                                                       pos_weight=ms.Tensor([1.0], ms.float32), reduction='none')

        p_t = preds * labels + (1 - preds) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
            loss = alpha_t * loss

        return loss.mean()
    
class loss_dice(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, preds, labels, eps=1.0):
        # labels = labels.squeeze(1).permute(1, 2, 0)
        # labels = self.resize_fn(labels.numpy()).transpose(2, 0, 1)
        # labels = ms.Tensor(labels, ms.float32)
        # preds = preds.squeeze(1)
        
        preds = ops.sigmoid(preds).flatten(start_dim=1)
        labels = labels.flatten(start_dim=1)
        numerater = 2.0 * (preds * labels).sum(1)
        # numerater = ops.mul(ops.sum(ops.mul(preds, labels), dim=1), 2.0)
        denominator = ops.add(preds.sum(1), labels.sum(1))
        denominator = ops.add(denominator, eps)
        numerater  = ops.add(numerater, eps)
        loss = 1.0 - numerater / denominator
        return loss.mean()