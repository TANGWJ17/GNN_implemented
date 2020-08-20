#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : layers.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 10:37   TANG       1.0         several layers
"""
import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as f

class Temporal_conv(nn.Module):
    """
    Temporal conv layer is a fundamental conv layer residually connected with a GLU
    Arg:
        KT: the kernel size of time axis
        KF: the kernel size of feature axis
        act_fun: the activate function ['GLU', 'linear', 'relu', 'sigmoid']
    Shape:
        Input: [batch_size, channels, frames, num_nodes, num_features]
        Kernel_size: [K_frames, K_nodes, k_features]
        Output: [batch_size, out_channels, frame_out, node_out, feature_out]
    """

    def __init__(self, in_channels, out_channels, KT, KF, act_fun='GLU'):
        super(Temporal_conv, self).__init__()
        self.KT = KT
        self.KF = KF
        self.act_fun = act_fun
        self.c_in = in_channels
        self.c_out = out_channels
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(KT, 1, KF))

    def forward(self, x):
        if len(x.shape) < 5:
            self.c_in = 1
            batch, T, N, F = x.shape
            x = x.reshape(batch, 1, T, N, F)
        else:
            batch, self.c_in, T, N, F = x.shape
        if self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x = torch.cat((x, torch.zeros(batch, self.c_out-self.c_in, T, N, F)), 1)
        elif self.c_in > self.c_out:
            # bottleneck down-sampling
            kernel = (1, 1, 1)
            x = nn.Conv3d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=kernel)(x)
        else:
            pass

        # keep the original input for residual connection.
        x_input = x[:, :, self.KT - 1:T, :, self.KF - 1:F]

        if self.act_fun is 'GLU':
            pass
        else:
            pass
            if self.act_fun is 'linear':
                pass
            elif self.act_fun is 'relu':
                pass
            elif self.act_fun is 'sigmoid':
                pass
            else:
                raise ValueError(f'ERROR: activation function "{self.act_fun}" is not defined.')


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
        if act_fun is 'GLU':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=(KT, 1))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(KT, 1))

    def forward(self, x):
        batch, self.c_in, T, N, F = x.shape

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

        if self.act_fun is 'GLU':
            return (x_conv[:, 0:self.c_out, :, :] + x_input) * f.sigmoid(x_conv[:, -self.c_out, :, :])
        else:
            if self.act_fun is 'linear':
                return x_conv
            elif self.act_fun is 'relu':
                return f.relu(x_conv)
            elif self.act_fun is 'sigmoid':
                return f.sigmoid(x_conv)
            else:
                raise ValueError(f'ERROR: activation function "{self.act_fun}" is not defined.')

# define message function and reduce function
def gcn_message(edges):
    # This computes a (batch of) message called 'msg' using the source node's feature 'feats' and learnable weigths.
    return {'msg': edges.src['feats'] * edges.data['weights']}  # edges.src.data is the attribute named ‘feats’


def gcn_reduce(nodes):
    # This computes the new 'feats' features by summing received 'msg' in each node's mailbox.
    return {'feats' : torch.sum(nodes.mailbox['msg'], dim=1)}

def graph_create(Y, infos, weights=None):
    """
    :param Y: [num_nodes, num_nodes] the processed Node admittance matrix
    :param infos: [num_nodes] the feature of each node
    :param weights: learnable weigths for message passing, using nn.Embedding()
    :return: update in orignal input
    """
    graph = dgl.DGLGraph(nx.from_numpy_matrix(Y))
    # add features to all the nodes
    # for i, x in enumerate(infos):
    #     graph.nodes[i].data['feats'] = x
    graph.ndata['feats'] = infos
    graph.edata['weights'] = Y * weights

    return graph

class Spatial_conv(nn.Module):
    """
        Spatial conv layer is a conv layer using message passing, achieved by DGL(deep graph library)
        each message passing operation deals with N nodes with one feature, so we need parallel processing
        Arg:
            Y: the processed Node admittance matrix
        Shape:
            Y: [batch_size, frames, num_nodes, num_nodes]
            Input: [batch_size, in_channels, frames, num_nodes]
            Output: [batch_size, out_channels, frame, num_nodes]
        """
    def __init__(self, in_channels, num_nodes):
        super(Spatial_conv, self).__init__()
        self.c_in = in_channels
        self.weights = nn.Embedding(in_channels, num_nodes, num_nodes)

    def forward(self, Y, x):
        # message passing
        for i in range(self.c_in):
            graph = graph_create(Y, x[:, i, :, :], self.weights[i, :, :])
            graph.send(graph.edges(), gcn_message)  # Trigger transmits information on all sides
            graph.recv(graph.nodes(), gcn_reduce)  # Trigger aggregation information on all sides
            # feats = graph.ndata.pop('feats')

        return f.relu(x)



if __name__ == '__main__':
    # xx = torch.randn(5, 33, 10, 20)
    # xx = xx.reshape((5, 1, 33, 10, 20))
    # conv = nn.Conv3d(1, 7, kernel_size=(4, 1, 20))
    # x_out = conv(xx)
    # print(x_out.shape)
    Y = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    info = np.array([10, 7, 1]).reshape(3, 1)
    a = graph_create(Y, info)
    print(a.ndata['feats'])
    a.nodes[1].data['feats'] = 1
    print(info)


