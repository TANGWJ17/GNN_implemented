#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : baseline.py
@Modify Time      @Author    @Version    @Desciption
2020/8/18 21:20   TANG       1.0         model based on STGCN: https://github.com/VeritasYin/STGCN_IJCAI-18
"""
import torch
import torch.nn as nn
import numpy as np
import dgl
from layers import Temporal_conv_layer, Spatial_conv, output_layer


class st_conv_block(nn.Module):
    """
        st_conv_block is a block consisted of 2 temporal layer and 1 spatial layer
        Arg:
            KT: the kernel size of time axis
            act_fun: the activate function ['GLU', 'linear', 'relu', 'sigmoid']
        Shape:
            Input: [batch_size, in_channels(num_features), frames, num_nodes]
            Kernel_size: [K_frames, K_nodes]
            Output: [batch_size, out_channels, frame_out, num_nodes]
        """
    def __init__(self, in_channels, out_channels_1, out_channels_2, KT,
                 num_nodes, batch_size, frames, frames_0):
        super(st_conv_block, self).__init__()
        self.frames_0 = frames_0
        self.KT = KT
        self.embed_1 = torch.nn.Parameter(torch.randn([in_channels, 1, num_nodes]))
        self.embed_2 = torch.nn.Parameter(torch.randn([out_channels_1, 1, num_nodes]))
        self.tempo_1_1 = Temporal_conv_layer(in_channels, out_channels_1, KT, act_fun='GLU')
        self.tempo_1_2 = Temporal_conv_layer(in_channels, out_channels_1, KT, act_fun='GLU')
        self.spat = Spatial_conv(out_channels_1, batch_size, frames - 2 * (KT - 1), num_nodes)
        self.tempo_2_1 = Temporal_conv_layer(out_channels_1, out_channels_2, KT, act_fun='GLU')
        self.tempo_2_2 = Temporal_conv_layer(out_channels_1, out_channels_2, KT, act_fun='GLU')

    def forward(self, infos, Y):
        x_0 = self.embed_1 * (infos[:, :, 0, :])[:, :, np.newaxis, :]
        x_1 = self.tempo_1_1(infos[:, :, 1:self.frames_0, :])
        x_2 = self.tempo_1_2(infos[:, :, self.frames_0:, :])
        x_conv = torch.cat([x_0, x_1, x_2], 2)
        x_updated = self.spat(Y, x_conv)
        g_0 = self.embed_2 * (x_updated[:, :, 0, :])[:, :, np.newaxis, :]
        g_1 = self.tempo_2_1(x_updated[:, :, 1:self.frames_0 - (self.KT - 1), :])
        g_2 = self.tempo_2_2(x_updated[:, :, self.frames_0 - (self.KT - 1):, :])
        g_conv = torch.cat([g_0, g_1, g_2], 2)
        return g_conv

class Baseline(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, KT_1, KT_2,
                 num_nodes, batch_size, frames, frames_0, num_generator):
        super(Baseline, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.st_1 = st_conv_block(in_channels, out_channels_1, out_channels_2, KT_1,
                 num_nodes, batch_size, frames, frames_0)
        self.st_2 = st_conv_block(out_channels_2, out_channels_2, out_channels_2, KT_2,
                 num_nodes, batch_size, frames - 4 * (KT_1 - 1), frames_0 - 2 * (KT_1 - 1))
        self.output_layer = output_layer(out_channels_2, frames - 4 * (KT_1 + KT_2 - 2), num_nodes, num_generator)

    def forward(self, Y, infos):
        infos_ = self.dropout(infos)
        x_1 = self.st_1(infos_, Y)
        x_2 = self.st_2(x_1, Y)
        return self.output_layer(x_2)

if __name__ == '__main__':
    print('-----')
 
