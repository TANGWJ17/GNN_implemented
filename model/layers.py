#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : layers.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 10:37   TANG       1.0         several layers
"""
import multiprocessing
from functools import partial

import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as f

from multiprocessing.dummy import Pool
np.set_printoptions(threshold=10000)

# class Temporal_conv(nn.Module):
#     """
#     Temporal conv layer is a fundamental conv layer residually connected with a GLU
#     Arg:
#         KT: the kernel size of time axis
#         KF: the kernel size of feature axis
#         act_fun: the activate function ['GLU', 'linear', 'relu', 'sigmoid']
#     Shape:
#         Input: [batch_size, channels, frames, num_nodes, num_features]
#         Kernel_size: [K_frames, K_nodes, k_features]
#         Output: [batch_size, out_channels, frame_out, node_out, feature_out]
#     """
#
#     def __init__(self, in_channels, out_channels, KT, KF, act_fun='GLU'):
#         super(Temporal_conv, self).__init__()
#         self.KT = KT
#         self.KF = KF
#         self.act_fun = act_fun
#         self.c_in = in_channels
#         self.c_out = out_channels
#         self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(KT, 1, KF))
#
#     def forward(self, x):
#         if len(x.shape) < 5:
#             self.c_in = 1
#             batch, T, N, F = x.shape
#             x = x.reshape(batch, 1, T, N, F)
#         else:
#             batch, self.c_in, T, N, F = x.shape
#         if self.c_in < self.c_out:
#             # if the size of input channel is less than the output,
#             # padding x to the same size of output channel.
#             x = torch.cat((x, torch.zeros(batch, self.c_out-self.c_in, T, N, F)), 1)
#         elif self.c_in > self.c_out:
#             # bottleneck down-sampling
#             kernel = (1, 1, 1)
#             x = nn.Conv3d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=kernel)(x)
#         else:
#             pass
#
#         # keep the original input for residual connection.
#         x_input = x[:, :, self.KT - 1:T, :, self.KF - 1:F]
#
#         if self.act_fun is 'GLU':
#             pass
#         else:
#             pass
#             if self.act_fun is 'linear':
#                 pass
#             elif self.act_fun is 'relu':
#                 pass
#             elif self.act_fun is 'sigmoid':
#                 pass
#             else:
#                 raise ValueError(f'ERROR: activation function "{self.act_fun}" is not defined.')


class Temporal_conv_layer(nn.Module):
    """
    Temporal conv layer is a fundamental conv layer residually connected with a GLU
    each feature is regarded as a single channel
    Arg:
        KT: the kernel size of time axis
        act_fun: the activate function ['GLU', 'linear', 'relu', 'sigmoid']
    Shape:
        Input: [batch_size, in_channels(num_features), frames, num_nodes]
        Kernel_size: [K_frames, K_nodes]
        Output: [batch_size, out_channels, frame_out, num_nodes]
    """

    def __init__(self, in_channels, out_channels, KT, act_fun='GLU'):
        super(Temporal_conv_layer, self).__init__()
        self.KT = KT
        self.act_fun = act_fun
        self.c_in = in_channels
        self.c_out = out_channels
        self.bh = nn.BatchNorm2d(in_channels)
        if act_fun is 'GLU':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=(KT, 1))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(KT, 1))

    def forward(self, x):
        batch, self.c_in, T, N = x.shape
        if self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x = torch.cat((x, torch.zeros(batch, self.c_out - self.c_in, T, N)), 1)
        elif self.c_in > self.c_out:
            # bottleneck down-sampling
            kernel = (1, 1)
            x = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=kernel)(x)
        else:
            pass
        # keep the original input for residual connection.
        x_input = x[:, :, self.KT - 1:T, :]
        x_conv = self.conv(x)
        # x_conv = self.bh(x_conv)

        if self.act_fun is 'GLU':

            return (x_conv[:, 0:self.c_out, :, :] + x_input) * torch.sigmoid(x_conv[:, -self.c_out:, :, :])
        else:
            if self.act_fun is 'linear':
                return x_conv
            elif self.act_fun is 'relu':
                return f.relu(x_conv)
            elif self.act_fun is 'sigmoid':
                return f.sigmoid(x_conv)
            else:
                raise ValueError(f'ERROR: activation function "{self.act_fun}" is not defined.')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)  # linear transformaer y=Wx+b
        self.activation = activation

    def forward(self, node):
        feats = self.linear(node.data['feats'])
        feats = self.activation(feats)
        return {'feats' : feats}   # return updated node feature feats(l+1)


def gcn_reduce(nodes):
    # This computes the new 'feats' features by summing received 'msg' in each node's mailbox.
    return {'feats': torch.mean(nodes.mailbox['msg'], dim=1)}
    # return dgl.function.mean('msg', 'feats')


class Spatial_conv(nn.Module):
    """
        Spatial conv layer is a conv layer using message passing, achieved by DGL(deep graph library)
        each message passing operation deals with N nodes with one feature, so we need parallel processing
        Arg:
            Y: the processed Node admittance matrix
        Shape:
            Y: [batch_size, frames, num_nodes, num_nodes]
            infos: [batch_size, in_channels, frames, num_nodes]
            Output: [batch_size, out_channels, frame, num_nodes]
        """
    def __init__(self, in_channels, batch_size, frames, num_nodes, Y=None, infos=None):
        super(Spatial_conv, self).__init__()
        self.c_in = in_channels
        self.Y = Y
        self.frames = frames
        self.num_nodes = num_nodes
        self.infos = infos
        self.graph_list = list(range(batch_size * frames))  # create a graph list for dgl batch
        self.apply_mod = NodeApplyModule(in_channels, in_channels, f.relu)
        # self.weights = nn.Embedding(in_channels, num_nodes, num_nodes)
        # self.weights = torch.nn.Parameter(torch.randn([in_channels, num_nodes, num_nodes]))
        self.denseLayer = torch.nn.Linear(3 * self.c_in, self.c_in)

    # define message function and reduce function
    def gcn_message(self, edges):
        # This computes a (batch of) message called 'msg' using the source node's feature 'feats' and learnable weigths.
        src_feats = edges.src['feats']
        dst_feats = edges.dst['feats']
        msg = edges.src['feats'] * self.denseLayer(torch.cat([src_feats, dst_feats, src_feats - dst_feats], 1))
        return {'msg': msg}  # edges.src.data is the attribute named ‘feats’

    def forward(self, Y, infos):
        # message passing using parallel processing
        batches, features, frames, nodes = infos.shape
        # x_conv = torch.zeros((batches, features, frames, nodes))
        self.Y = Y
        self.infos = infos
        # not use a parallel processing
        # for i in range(batches * frames):
        #     self.graph_update(i)
        # using pool to accelerate the process
        pool = Pool()
        pool.map(self.graph_update, range(batches * frames))
        pool.close()
        # create a graph batch and do message passing
        bg_graph = dgl.batch(self.graph_list)
        # bg_graph.send(bg_graph.edges(), self.gcn_message)  # Trigger transmits information on all sides
        # bg_graph.recv(bg_graph.nodes(), self.gcn_reduce)  # Trigger aggregation information on all sides
        bg_graph.update_all(self.gcn_message, gcn_reduce)
        bg_graph.apply_nodes(func=self.apply_mod)
        # get the updated features
        # todo use unbatch to get the ndata
        updated_x = bg_graph.ndata['feats'].reshape((batches, frames, nodes, features))
        updated_x = updated_x.permute(0, 3, 1, 2)
        return f.relu(updated_x)

    def graph_update(self, number):
        """
        :param Y: [batch_size, frames, num_nodes, num_nodes] the processed Node admittance matrix
        :param infos: [batch_size, in_channels, frames, num_nodes] the features of each node
        :param weights: [features, num_nodes, num_nodes] learnable weigths for message passing, using nn.Embedding()
        :return: graph
        """
        # todo frames的判断应该根据frame_0来
        batches = number // self.frames
        frames = number % self.frames
        Y_number = 0
        if frames >= 1:
            Y_number = 1
            if frames > 11:
                Y_number = 2
        Y_need = self.Y[batches, Y_number, :, :].cpu().numpy()
        nx_graph = nx.from_numpy_matrix(Y_need)
        graph = dgl.from_networkx(nx_graph).to(torch.device('cuda:0'))
        # add features to all the nodes
        graph.ndata['feats'] = (self.infos[batches, :, frames, :]).T
        # graph.edata['weights'] = (self.Y[batches, Y_number, :, :] *
        #                                       self.weights).permute((1, 2, 0))\
        #     .reshape((self.num_nodes * self.num_nodes, self.c_in))

        self.graph_list[number] = graph
        return graph


class output_layer(nn.Module):
    def __init__(self, features, frames, num_nodes, num_generator):
        super(output_layer, self).__init__()
        self.num_generator = num_generator
        self.output = nn.Linear(features * frames * num_nodes, num_generator)

    def forward(self, inputs):
        batches, features, frames, nodes = inputs.shape
        output = self.output(inputs.contiguous().view(batches, -1))
        return torch.sigmoid(output.view(batches, self.num_generator))

class Spatial_attention_layer(nn.Module):
    def __init__(self, feats, frames, nodes):
        super(Spatial_attention_layer, self).__init__()
        self.W_1 = torch.randn((frames, ))
        self.W_2 = torch.randn((feats, frames))
        self.W_3 = torch.randn((feats, ))
        self.b_s = torch.randn((1, nodes, nodes))
        self.V_s = torch.randn((nodes, nodes))

    def forward(self, x):
        # x = x.transpose((0, 2, 3, 1))
        # compute spatial attention scores
        # shape of lhs is (batch_size, nodes, frames)
        lhs = (x @ self.W_1) @ self.W_2

        # shape of rhs is (batch_size, frames, nodes)
        rhs = self.W_3 @ x.transpose((2, 0, 3, 1))

        # shape of product is (batch_size, nodes, nodes)
        product = torch.matmul(lhs, rhs)

        S = (self.V_s @ f.sigmoid(product + self.b_s).transpose((1, 2, 0))).transpose((2, 0, 1))

        # normalization
        S = S - torch.max(S, dim=1)
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, dim=1)
        return S_normalized

class cheb_conv_SAT(nn.Module):
    def __init__(self, filters, features, K, cheb_polynomials):
        super(cheb_conv_SAT, self).__init__()
        self.K = K
        self.filters = filters
        self.Theta = torch.randn((K, features, filters))
        self.cheb_polynomials = cheb_polynomials

    def forward(self, x, spatial_attention):
        batches, nodes, features, frame = x.shape
        outputs = []
        for time_step in range(frame):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batches, nodes, self.filters))
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                rhs = torch.matmul(T_k_with_at.transpose((0, 2, 1)),
                                   graph_signal)

                output = output + rhs @  theta_k
            outputs.append(output.expand_dims(-1))
        return f.relu(torch.cat(*outputs, dim=-1))

class Temporal_attention_layer(nn.Module):
    def __init__(self, features, frames, nodes):
        super(Temporal_attention_layer, self).__init__()
        self.U_1 = torch.randn((nodes, ))
        self.U_2 = torch.randn((features, nodes))
        self.U_3 = torch.randn((features, ))
        self.b_e = torch.randn((1, frames, frames))
        self.V_e = torch.randn((frames, frames))

    def forward(self, x):
        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = (x.transpose((0, 3, 2, 1)) @ self.U_1) @ self.U_2

        # shape is (N, V, T)
        rhs = self.U_3 @  x.transpose((2, 0, 1, 3))

        product = torch.matmul(lhs, rhs)

        E = self.V_e @ f.sigmoid(product + self.b_e).transpose((1, 2, 0)).transpose((2, 0, 1))

        # normailzation
        E = E - torch.max(E, dim=1)
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, dim=1)
        return E_normalized

class ASTGCN_block(nn.Module):
    def __init__(self, backbone):
        super(ASTGCN_block, self).__init__()
        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]
        features = backbone['features']
        frames = backbone['frames']
        nodes = backbone['nodes']
        self.SAT = Spatial_attention_layer(features, frames, nodes)
        self.cheb_conv = cheb_conv_SAT(num_of_chev_filters, features, K, cheb_polynomials)
        self.TAT = Temporal_attention_layer(features, frames, nodes)
        self.time_conv = nn.Conv2d(in_channels=features, out_channels=num_of_time_filters, kernel_size=(1, 3))
        self.residual_conv = nn.Conv2d(in_channels=features, out_channels=num_of_time_filters, kernel_size=(1, 1))
        self.ln = nn.LayerNorm() # TODO set the size of layer

    def forward(self, x):
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        # shape is (batch_size, T, T)
        temporal_At = self.TAt(x)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps),
                             temporal_At) \
            .reshape(batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        # cheb gcn with spatial attention
        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        # convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.transpose((0, 2, 1, 3))).transpose((0, 2, 1, 3))

        # residual shortcut
        x_residual = (self.residual_conv(x.transpose((0, 2, 1, 3)))
                      .transpose((0, 2, 1, 3)))

        return self.ln(f.relu(x_residual + time_conv_output))

if __name__ == '__main__':
    # xx = torch.randn(5, 33, 10, 20)
    # xx = xx.reshape((5, 1, 33, 10, 20))
    # conv = nn.Conv3d(1, 7, kernel_size=(4, 1, 20))
    # x_out = conv(xx)
    # print(x_out.shape)

    # Y = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    # info = np.array([10, 7, 1]) # .reshape(3, 1)
    # a = graph_create(Y, info)
    # print(a.ndata['feats'])
    data = torch.randn((10, 6, 33, 9))
    out = nn.Linear(6 * 33 * 9, 3)
    output = out(data.view(10, -1))
    print(output.shape)
    outout = f.relu(output.view(10, 3))
    print(outout)





