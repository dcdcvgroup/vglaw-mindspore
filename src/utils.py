import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import math


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = ms.Parameter(ops.ones(normalized_shape))
        self.bias = ms.Parameter(ops.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    # @ms.jit
    def construct(self, x):
        u = x.mean(axis=1, keep_dims=True)
        s = (x - u).pow(2).mean(axis=1, keep_dims=True)
        x = (x - u) / ops.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x +self.bias[:, None, None]

        return x


def box_ious(boxes1, boxes2, types='iou'):
    """
    Calculate the ious of two boxes array.

    Args:
        boxes1(Tensor): a list of boxes, (N,(cx,cy,w,h)).
        boxes2(Tensor): a list of boxes, (N,(cx,cy,w,h)).
        types(str): the types of iou.
    """

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
    intersect_rb = ops.minimum(b1_rb, b2_rb)  # (N,(x,y))
    dif = intersect_rb - intersect_lt  # (N,(dif_x,dif_y))
    clip_max_value = 10
    intersect_wh = ops.clip_by_value(dif, clip_value_min=0, clip_value_max=clip_max_value)  # (N,(w,h))
    intersect_area = ops.prod(intersect_wh, axis=1)  # (N,area)
    union_area = ops.prod(boxes1[:, 2:], 1) + ops.prod(boxes2[:, 2:], 1) - intersect_area  # (N,area)
    clip_max_value = 10
    # iou = intersect_area / union_area.clamp(min=1e-6)  # N
    iou = intersect_area / \
        ops.clip_by_value(union_area, clip_value_min=1e-6,
                          clip_value_max=clip_max_value)
    results['iou'] = iou
    # dependence: giou, diou, ciou

    # init parameters to avoid bug in graph mode
    enclose_lt = 0
    enclose_rb = 0
    enclose_wh = 0
    enclose_area = 0
    diou = 0
    if 'giou' in types or 'diou' in types or 'ciou' in types:
        enclose_lt = ops.min(ops.stack([b1_lt, b2_lt]))[1]
        enclose_rb = ops.max(ops.stack([b1_rb, b2_rb]))[1]
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


def box_ious_v2(boxes1, boxes2, types='iou'):
    """
    Calculate the ious of two boxes array.

    Args:
        boxes1(Tensor): a list of boxes, (N,(cx,cy,w,h)).
        boxes2(Tensor): a list of boxes, (N,(cx,cy,w,h)).
        types(str): the types of iou.
    """

    types = [types] if isinstance(types, str) else types
    results = {}
    boxes2=boxes2.astype(boxes1.dtype)

    b1_cxcy = boxes1[:, :2]  # center x and center y of boxes1, (N,(cx,cy))  32, 2
    b2_cxcy = boxes2[:, :2]  # center x and center y of boxes2, (N,(cx,cy))  32, 2
    # half of the width and height of boxes1, (N,(0.5w,0.5h))
    b1_wh_half = 0.5 * boxes1[:, 2:]  # 32, 2
    # half of the width and height of boxes2, (N,(0.5w,0.5h))
    b2_wh_half = 0.5 * boxes2[:, 2:]  # 32, 2

    b1_lt = b1_cxcy - b1_wh_half  # left top of boxes1, (N,(lt_x,lt_y))  32, 2
    b1_rb = b1_cxcy + b1_wh_half  # right bottom of boxes1, (N,(rb_x,rb_y))
    b2_lt = b2_cxcy - b2_wh_half  # left top of boxes2, (N,(lt_x,lt_y))
    b2_rb = b2_cxcy + b2_wh_half  # right bottom of boxes2, (N,(rb_x,rb_y))

    # dependence: iou, giou, diou, ciou
    intersect_lt = ops.maximum(b1_lt, b2_lt)  # (N,(x,y))
    intersect_rb = ops.minimum(b1_rb, b2_rb)  # (N,(x,y))
    dif = intersect_rb - intersect_lt  # (N,(dif_x,dif_y))
    clip_max_value = 10
    intersect_wh = ops.clip_by_value(dif, clip_value_min=0, clip_value_max=clip_max_value)  # (N,(w,h))
    intersect_area = ops.prod(intersect_wh, axis=1)  # (N,area)
    union_area = ops.prod(boxes1[:, 2:], 1) + ops.prod(boxes2[:, 2:], 1) - intersect_area  # (N,area)
    clip_max_value = 10
    # iou = intersect_area / union_area.clamp(min=1e-6)  # N
    iou = intersect_area / \
        ops.clip_by_value(union_area, clip_value_min=1e-6,
                          clip_value_max=clip_max_value)
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
        enclose_wh = ops.sub(enclose_rb, enclose_lt)  # N,2
        enclose_area = ops.prod(enclose_wh, 1)  # N
        area_sub = ops.sub(enclose_area, union_area)
        area_div = ops.div(area_sub, enclose_area)
        giou = ops.sub(iou, area_div)
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


def noam_warmup(step, gamma, warmup_steps):
    step = step + 1
    return min(step ** gamma, step * warmup_steps ** (gamma - 1)) * warmup_steps ** -gamma


def mask_iou(mask_gt, mask_pred, eps=1e-10):
    mask_gt, mask_pred = mask_gt.flatten(start_dim=1).gt(0), mask_pred.flatten(start_dim=1).gt(0)
    intersection = ops.logical_and(mask_gt, mask_pred).float()
    union = ops.logical_or(mask_gt, mask_pred).float()
    iou = (ops.sum(intersection, 1) + eps) / (ops.sum(union, 1) + eps)

    ap = dict()
    thresholds = ops.arange(0.5, 1, 0.05)

    for threshold in thresholds:
        ap[ms.Tensor(threshold)] = iou.gt(threshold).float()

    return iou, ap


def dice_loss(preds, targets, eps=1.0):
    preds = ops.sigmoid(preds).flatten(start_dim=1)
    targets = ops.flatten(targets, start_dim=1)
    numerater = 2 * (preds * targets).sum(1)
    denominator = preds.sum(1) + targets.sum(1)
    loss = 1 - (numerater + eps) / (denominator + eps)
    return loss.mean()


def sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2):
    labels = targets.flatten(start_dim=0)
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

