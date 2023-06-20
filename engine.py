# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import numpy as np
from util import box_ops

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Checking if learns to predict spindles

    #probs = outputs['pred_logits'].sigmoid()
    #probs = probs.cpu().detach()
    #probs = np.asarray(probs)

    #print(np.amax(probs, axis = 1))

    F1, TP, total_pred_count, total_spindle_count = f1_calculate(model, device, data_loader)
    row = {'F1': F1, 'TP': TP, 'Total pred': total_pred_count, 'Total spindle': total_spindle_count}

    return row, {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        #orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    #probs = outputs['pred_logits'].sigmoid()
    #probs = probs.cpu().detach()
    #probs = np.asarray(probs)

    #print(np.amax(probs, axis = 1))


    F1, TP, total_pred_count, total_spindle_count = f1_calculate(model, device, data_loader)
    row = {'F1': F1, 'TP': TP, 'Total pred': total_pred_count, 'Total spindle': total_spindle_count}


   
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return row, stats, coco_evaluator

def pred_stats(outputs, targets):
    
    # Loop through batches to compute F1 score through training.

    
    F1_list = []
    temp_tp = 0
    total_spindle_count = 0
    total_pred_count = 0

    out_logits = outputs['pred_logits']
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 3, dim=1)
    scores = topk_values
    print(torch.mean(scores, 0))
    topk_boxes = topk_indexes // out_logits.shape[2]
        
    boxes = outputs['pred_boxes'][:,:,:].detach().cpu().numpy()
    topk_indexes = topk_boxes.cpu().numpy()
    #print(topk_indexes.shape)
    #print(boxes.shape)


    #print(scores)
    #print(topk_boxes[0,0])
    #print(outputs['pred_boxes'][0,topk_boxes[0,0]])
    for i in range(topk_indexes.shape[0]):
        idxs = topk_indexes[i,:]
        kept_boxes = boxes[i, idxs]
        target_bbox = targets[i]['boxes']
        
        TP = 0
        
        target_bbox = target_bbox.cpu()
        target_bbox = target_bbox.numpy()
        total_spindle_count += target_bbox.shape[0]
        total_pred_count += len(kept_boxes)
        for k in range(target_bbox.shape[0]):
            tar_box = target_bbox[k,:]
            tar_box_start = tar_box[0] - tar_box[1]/2
            tar_box_end = tar_box[0] + tar_box[1]/2
            
            best_match = -1
            
            if len(kept_boxes) == 0:
                continue
            
            for j,out_box in enumerate(kept_boxes):
                out_box_start = out_box[0] - out_box[1]/2
                out_box_end = out_box[0] + out_box[1]/2

                if overlap(out_box, tar_box) > overlap(kept_boxes[best_match], tar_box):
                    best_match = j
            
            if overlap(kept_boxes[best_match],tar_box) > 0.2:
                TP +=1            

        # FP = total_pred_count - TP
        # FN = total_spindle_count - TP
        
        # if (TP + FP) == 0:
        #     PRECISION = TP
        # else:
        #     PRECISION = (TP)/(TP + FP)
        
        # RECALL = (TP)/(TP+FN)

        # if (PRECISION + RECALL) == 0:
        #     F1_list.append(0)
        # else:
        #     F1_list.append((2 * PRECISION * RECALL)/(PRECISION + RECALL))
        
        temp_tp += TP
    #print(kept_boxes)

    #F1_list = np.asarray(F1_list)
    #print("F1 MEAN:", np.mean(F1_list), " F1 STD:", np.std(F1_list), " TP:", temp_tp, " FP:", FP, " Number of spindles:", total_spindle_count)
    return (temp_tp, total_pred_count, total_spindle_count)

def f1_calculate(model, device, dataloader):
    TP = 0
    total_pred_count = 0
    total_spindle_count = 0
    for samples, targets in dataloader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        temp_tp, temp_pred_count, temp_spindle_count = pred_stats(outputs, targets)
        TP += temp_tp
        total_pred_count += temp_pred_count
        total_spindle_count += temp_spindle_count
    
    f1 = f1_score(TP, total_pred_count, total_spindle_count)

    print("F1 score:", f1, " True positives:", TP, " Total predictions:", total_pred_count, " Total spindles:", total_spindle_count)

    return (f1, TP, total_pred_count, total_spindle_count)
        


def f1_score(TP, total_pred_count, total_spindle_count):
    
    FP = total_pred_count - TP
    FN = total_spindle_count - TP
        
    if (TP + FP) == 0:
        PRECISION = TP
    else:
        PRECISION = (TP)/(TP + FP)
        
    RECALL = (TP)/(TP+FN)

    if (PRECISION + RECALL) == 0:
            F1 = 0
    else:
         F1 = (2 * PRECISION * RECALL)/(PRECISION + RECALL)

    return F1



def iou(out,tar):
    out_box_start = out[0] - out[1]/2
    out_box_end = out[0] + out[1]/2

    tar_box_start = tar[0] - tar[1]/2
    tar_box_end = tar[0] + tar[1]/2

    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    return ((overlap_end - overlap_start)/(union_end-union_start))

def overlap(out, tar):
    out_box_start = out[0] - out[1]/2
    out_box_end = out[0] + out[1]/2

    tar_box_start = tar[0] - tar[1]/2
    tar_box_end = tar[0] + tar[1]/2

    #print(out)
    #print(tar)
    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    return (overlap_end - overlap_start) / ((tar_box_end-tar_box_start))