# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
    

    
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        n_groups = 8
        
        # ENCODER GROUND LEVEL (LEVEL 1)
        self.conv_1_1 = nn.Conv1d(1, 32, kernel_size = 5, dilation = 2)
        self.batch_1_1 = nn.BatchNorm1d(32)

        self.conv_1_2 = nn.Conv1d(32, 32, kernel_size = 5, dilation = 2)
        self.batch_1_2 = nn.BatchNorm1d(32)


        # ENCODER BOTTOM LEVEL
        self.pool_1 = nn.MaxPool1d(kernel_size = 2)

        self.conv_2_1 = nn.Conv1d(32, 64, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2_1 = nn.BatchNorm1d(64)

        self.conv_2_2 = nn.Conv1d(64, 64, kernel_size = 5, dilation = 2, padding = 'same')
        self.batch_2_2 = nn.BatchNorm1d(64)

        
        # DECODER GROUND LEVEL (LEVEL 1)
        # UPSAMPLING
        self.upsample_1 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv_1_3 = nn.Conv1d(64, 32, kernel_size = 2, dilation = 1, padding = 'same')

        # 
        self.conv_1_4 = nn.Conv1d(64, 32, kernel_size = 5, dilation = 1)
        self.batch_1_4 = nn.BatchNorm1d(32)

        self.conv_1_5 = nn.Conv1d(32, 32, kernel_size = 5, dilation = 1)
        self.batch_1_5 = nn.BatchNorm1d(32)


        #self.conv_1 = nn.Conv1d(1, 64, kernel_size = 5, stride = 2, dilation = 2)
        #self.batch_1 = nn.BatchNorm1d(128)

        #self.conv_2 = nn.Conv1d(64, 128, kernel_size = 5, stride = 2, dilation = 2)
        #self.batch_2 = nn.BatchNorm1d(128)

        #self.conv_3 = nn.Conv1d(128, 256, kernel_size = 5, stride = 2, dilation = 2)
        #self.batch_3 = nn.BatchNorm1d(128)
        

        self.num_channels = 32



    def forward(self, tensor_list: NestedTensor):
        # GROUND LEVEL FORWARD
        level_1 = self.batch_1_1(F.relu(self.conv_1_1(tensor_list.tensors)))
        level_1 = self.batch_1_2(F.relu(self.conv_1_2(level_1)))

        # POOLING AND BOTTOM LEVEL
        level_1_down = self.pool_1(level_1)
        level_2 = self.batch_2_1(F.relu(self.conv_2_1(level_1_down)))
        level_2 = self.batch_2_2(F.relu(self.conv_2_2(level_2)))

        # UPSAMPLING AND FEATURE FUSION
        level_2_upsampled = self.upsample_1(level_2)
        level_1_up = self.conv_1_3(level_2_upsampled)
        
        dec_level_1 = torch.cat((level_1, level_1_up), 1)

        dec_level_1 = self.batch_1_4(F.relu(self.conv_1_4(dec_level_1)))
        dec_level_1 = self.batch_1_5(F.relu(self.conv_1_5(dec_level_1)))

        #print(x_1.shape)
        #print(x_3.shape)

        #features = torch.cat((x_1, x_2, x_3), 1)
        #features = self.pool_4(self.batch_4(F.relu(self.conv_4(features))))

        
        #print(x_3.shape)
        
        features = dec_level_1

        #xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        
        x = features[:,:,None,:]
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-1:]).to(torch.bool)[0]
        out['CNN_OUT'] = NestedTensor(x, mask)
        return out


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        n_groups = 8
        
        # ENCODER GROUND LEVEL (LEVEL 1)
        self.conv_1 = nn.Conv1d(1, 64, kernel_size = 3, dilation = 1)
        self.batch_1 = nn.BatchNorm1d(64)

        self.conv_2 = nn.Conv1d(64, 64, kernel_size = 3, dilation = 1)
        self.batch_2 = nn.BatchNorm1d(64)

        self.conv_3 = nn.Conv1d(64, 64, kernel_size = 3, dilation = 1)
        self.batch_3 = nn.BatchNorm1d(64)

        

        self.num_channels = 64



    def forward(self, tensor_list: NestedTensor):
        # GROUND LEVEL FORWARD
        features = self.batch_1(F.relu(self.conv_1(tensor_list.tensors)))
        features = self.batch_2(F.relu(self.conv_2(features)))
        features = self.batch_3(F.relu(self.conv_3(features)))

        
        #xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        
        x = features[:,:,None,:]
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-1:]).to(torch.bool)[0]
        out['CNN_OUT'] = NestedTensor(x, mask)
        return out


class building_block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size = 1, pool_k_size = 1,conv_padd = 0):
        super(building_block, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = k_size, padding = conv_padd)
        self.batch = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size = pool_k_size)

    def forward(self, x):
        output = self.pool(self.batch(F.relu(self.conv(x))))

        return output


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = args.masks
    #backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    backbone = Backbone()
    model = Joiner(backbone, position_embedding)
    #model.num_channels = backbone.num_channels
    model.num_channels = 64
    return model
