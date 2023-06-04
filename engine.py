# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import numpy as np
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    batch_counter = 0
    f1_mean_run = []
    f1_std_run = []
    TP_run = []
    FP_run = []
    total_spindle_run = []
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
        #if max_norm > 0:
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        #if (batch_counter + 1) % 8 == 0 or (batch_counter + 1 == len(data_loader)):
            #optimizer.step()
            #optimizer.zero_grad()
        #lr_scheduler.step()
        batch_counter += 1
        #f1_score(outputs, targets)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        f1_mean, f1_std, TP, FP, total_spindle_count = f1_score(outputs, targets)
        f1_mean_run.append(f1_mean)
        f1_std_run.append(f1_std)
        TP_run.append(TP)
        FP_run.append(FP)
        total_spindle_run.append(total_spindle_count)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print("Averaged stats:", metric_logger)

    print("F1 MEAN:", sum(f1_mean_run)/len(f1_mean_run), " F1 STD:", sum(f1_std_run)/len(f1_std_run), " TP:", sum(TP_run), " FP:", sum(FP_run),
     " Number of spindles:", sum(total_spindle_run))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
        #results = postprocessors['bbox'](outputs, orig_target_sizes)
        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        #res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name

        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator = None
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


def f1_score(outputs, targets):
    
    # Loop through batches to compute F1 score through training.

    
    F1_list = []
    temp_tp = 0
    total_spindle_count = 0
    for i in range(outputs['pred_logits'].shape[0]):
        probas = outputs['pred_logits'].softmax(-1)[i,:,:-1]
        keep = probas.max(-1).values > 0.7
        kept_boxes = outputs['pred_boxes'][i,keep]
        target_bbox = targets[i]['boxes']
        
        TP = 0
        total_pred_count = 0
        
        target_bbox = target_bbox.cpu()
        target_bbox = target_bbox.numpy()
        total_spindle_count += target_bbox.shape[0]
        total_pred_count += len(kept_boxes)
        for k in range(target_bbox.shape[0]):
            tar_box = target_bbox[k,:]
            tar_box_start = tar_box[0] - tar_box[1]/2
            tar_box_end = tar_box[0] + tar_box[1]/2
            
            best_match = -1
            
            for j,out_box in enumerate(kept_boxes):
                out_box_start = out_box[0] - out_box[1]/2
                out_box_end = out_box[0] + out_box[1]/2

                if ((out_box_end > tar_box_start) and (out_box_start <= tar_box_start)):
                    if iou(out_box, tar_box) > iou(kept_boxes[best_match], tar_box):
                        best_match = j
            if len(kept_boxes) == 0:
                continue
            if overlap(kept_boxes[best_match],tar_box,0.2):
                TP +=1
            

        FP = total_pred_count - TP
        FN = total_spindle_count - TP
        
        if (TP + FP) == 0:
            PRECISION = TP
        else:
            PRECISION = (TP)/(TP + FP)
        
        RECALL = (TP)/(TP+FN)

        if (PRECISION + RECALL) == 0:
            F1_list.append(0)
        else:
            F1_list.append((2 * PRECISION * RECALL)/(PRECISION + RECALL))
        
        temp_tp += TP


    F1_list = np.asarray(F1_list)
    #print("F1 MEAN:", np.mean(F1_list), " F1 STD:", np.std(F1_list), " TP:", temp_tp, " FP:", FP, " Number of spindles:", total_spindle_count)
    return (np.mean(F1_list), np.std(F1_list), temp_tp, FP, total_spindle_count)


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

def overlap(out, tar, threshold):
    out_box_start = out[0] - out[1]/2
    out_box_end = out[0] + out[1]/2

    tar_box_start = tar[0] - tar[1]/2
    tar_box_end = tar[0] + tar[1]/2

    overlap_start = max(out_box_start, tar_box_start)
    overlap_end = min(out_box_end, tar_box_end)
    union_start = min(out_box_start, tar_box_start)
    union_end = max(out_box_end, tar_box_end)

    if (overlap_end - overlap_start) >= (threshold * (tar_box_end-tar_box_start)):
        return True
    else:
        return False