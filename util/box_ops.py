# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, w = x.unbind(-1)
    b = [(x_c - 0.5 * w),(x_c + 0.5 * w)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, x1= x.unbind(-1)
    b = [(x0 + x1) / 2,(x1 - x0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    #area1 = box_area(boxes1)
    #area2 = box_area(boxes2)

    len1 = boxes1[:, 1] - boxes1[:,0]
    len2 = boxes2[:, 1] - boxes2[:, 0]

    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,2]

    #wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = (rb-lt).clamp(min=0)  # [N,M]

    union = len1[:, None] + len2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1] >= boxes1[:, 0]).all()
    assert (boxes2[:, 1] >= boxes2[:, 0]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    #wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = (rb-lt).clamp(min=0)

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
