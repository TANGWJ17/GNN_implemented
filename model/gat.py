#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : gat.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/24 8:52   TANG       1.0         model based on ASTGCN: https://github.com/wanhuaiyu/ASTGCN
"""
import torch
import torch.nn as nn
import dgl
import numpy as np
from layers import ASTGCN_block
# TODO to transpose the input matrix to [batch, nodes, features, frames]
class ASTGCN_submodule(nn.Module):
    """
        ASTGCN_submodule integrate the ASTGCN layers
        Arg:
            backbones: List of backbones
    """
    def __init__(self, backbones):
        super(ASTGCN_submodule, self).__init__()
        self.blocks = nn.Sequential()
        for i, backbone in enumerate(backbones):
            self.blocks.add_module('backbone_{}'.format(i), ASTGCN_block(backbone))
        self.final_conv = nn.Conv2d(out_channels=1, kernel_size=(1, backbones[-1]['num_of_time_filters']))

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

class GAT_edge(nn.Module):
    def __init__(self):
        super(GAT_edge, self).__init__()

 
