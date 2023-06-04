# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from dreams_dataloader import dreams_dataset

from pytorch_model_summary import summary
import torch.nn as nn

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr_drop', default=800, type=int)
    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=10, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=10, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=1, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    model, criterion, postprocessors = build_model(args)
    #print(summary(model(), torch.zeros((1, 1, 7680)), show_input=True))

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), " GPU's GIVEN")
            model = nn.DataParallel(model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if ("backbone" or "class_embed" or "bbox_embed" or "query") not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("backbone" or "class_embed" or "bbox_embed" or "query") in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    n_list = []
    for n, p in model_without_ddp.named_parameters():
        n_list.append(n)
    #print(n_list)

    #dataset = dreams_dataset()
    #train_size = int(len(dataset)* 0.9)
    #val_size = int(len(dataset) - train_size)
    #dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size,val_size])
    #dataset_val = dataset_train

    dataset_train = dreams_dataset(input_path = '/scratch/s174411/FIL_CEN/TRAIN/input/', label_path = '/scratch/s174411/FIL_CEN/TRAIN/labels/')
    dataset_val = dreams_dataset(input_path = '/scratch/s174411/FIL_CEN/VAL/input/', label_path = '/scratch/s174411/FIL_CEN/VAL/labels/')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)


    def custom_collate(original_batch):
        label_list = []
        food_list = []
        for item in original_batch:
            food, label = item
            label_list.append(label)
            food_list.append(food)

        food_list = torch.stack(food_list)
        food_list = food_list.float()
        return food_list,label_list
    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=custom_collate, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,drop_last=False, collate_fn=custom_collate, num_workers=args.num_workers)

    def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
        base_lr = 1
        factor = 0.01
        return base_lr/(1+factor*epoch)


    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.1)
    
    #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                    #max_lr = 1e-5, # Upper learning rate boundaries in the cycle for each parameter group
                    #steps_per_epoch = len(data_loader_train),
                    #epochs = 60,
                    #pct_start = 0.2,
                    #cycle_momentum=False)



    #scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6], gamma=10)
    #scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[55,,35,45], gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])


    base_ds = get_coco_api_from_dataset(dataset_val)

    for g in optimizer.param_groups:
        print(g['lr'])

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, lr_scheduler, device, epoch,
            args.clip_max_norm)
        #lr_scheduler.step()
        if epoch == 1000:
            # Lr transformer
            optimizer.param_groups[0]['lr'] = 1e-4
            # Lr backbone
            optimizer.param_groups[1]['lr'] = 1e-4
            print("LR of model changed to ", optimizer.param_groups[0]['lr'])
            print("LR of backbone changed to ", optimizer.param_groups[1]['lr'])

        if epoch == 1000:
            # Lr transformer
            optimizer.param_groups[0]['lr'] = 1e-5
            # Lr backbone
            optimizer.param_groups[1]['lr'] = 1e-5
            print("LR of model changed to ", optimizer.param_groups[0]['lr'])
            print("LR of backbone changed to ", optimizer.param_groups[1]['lr'])

        if epoch == 1000:
            # Lr transformer
            optimizer.param_groups[0]['lr'] = 1e-4
            # Lr backbone
            optimizer.param_groups[1]['lr'] = 1e-5
            print("LR of model changed to ", optimizer.param_groups[0]['lr'])
            print("LR of backbone changed to ", optimizer.param_groups[1]['lr'])

        if epoch == 1000:
            # Lr transformer
            optimizer.param_groups[0]['lr'] = 1e-5
            # Lr backbone
            optimizer.param_groups[1]['lr'] = 1e-6
            print("LR of model changed to ", optimizer.param_groups[0]['lr'])
            print("LR of backbone changed to ", optimizer.param_groups[1]['lr'])


        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
