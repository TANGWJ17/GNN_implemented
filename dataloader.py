#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : dataloader.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/18 21:16   TANG       1.0         dataloader for networks
"""

import os
import numpy as np
import torch
import torch.nn as nn
import dgl
from torch.utils.data import Dataset, DataLoader


def Y_process(Y_):
    row, col = np.diag_indices(Y_.shape[0])
    Y_[row, col] = np.zeros(Y_.shape[0])
    return Y_


def label_data(power_angle):
    abs_data = abs(power_angle)
    dMin, dMax = abs_data.min(), abs_data.max()
    angle_1 = abs(abs_data - dMin) > 180
    angle_2 = abs(abs_data - dMax) > 180
    angle_labeled = list(map(int, angle_1)) if np.sum(angle_1) < np.sum(angle_2) else list(map(int, angle_2))
    return np.array(angle_labeled)


def read_data(num_nodes, num, number_data):
    all_Data = np.load('./data/sample_0_{}.npz'.format(number_data), allow_pickle=True)
    Data = all_Data['arr_0']
    name_dict = dict()
    node_names = Data[0]
    num_generator = 10
    Y_RAW = np.load('./data/Y_raw.npz', allow_pickle=True)['arr_0']
    # for node_name in node_names:
    #     num_generator += 1 if '发电' in node_name else 0
    for i in range(len(node_names)):
        name_dict[node_names[i]] = i
    Y = np.zeros([num, 3, num_nodes, num_nodes])  # num * 3(before/ing/after) * num_nodes * num_nodes
    infos = np.zeros([num, 6 + 1, 33, num_nodes])  # num * feature(+1) * frames * num_nodes
    labels = np.zeros([num, num_generator])  # num * num_generator
    Y_raw = np.sqrt(np.power(Y_process(Y_RAW[0]), 2) + np.power(Y_process(Y_RAW[1]), 2))
    Y[:, 0, :, :] = Y_raw  # the original Y(before accident)
    for i in range(num_nodes):
        added = -2 if number_data > 0 else 0
        infos[i, 6, :, name_dict[Data[4 + i * 10 + added]]] = 1  # feature of trouble node number
        Y_in = np.sqrt(
            np.power(Y_process(Data[6 + i * 10 + added][0]), 2) + np.power(Y_process(Data[6 + i * 10 + added][1]), 2))
        Y[:, 1, :, :] = Y_in  # the Y(in accident)
        Y_after = np.sqrt(
            np.power(Y_process(Data[6 + i * 10 + added][2]), 2) + np.power(Y_process(Data[6 + i * 10 + added][3]), 2))
        Y[:, 2, :, :] = Y_after  # the Y(after accident)
        where_are_nan = np.isnan(Data[11 + i * 10 + added])
        Data[11 + i * 10 + added][where_are_nan] = 200
        # labels[i] = np.array(list(map(int, (abs(Data[11 + i * 10 + added]) - 180) > 0)))
        labels[i] = label_data(Data[11 + i * 10 + added])
        infos[i, :6, :, :] = np.transpose(Data[12 + i * 10 + added].reshape(33, num_nodes, 6), [2, 0, 1])
    for i in range(num_nodes, num):
        added = i // num_nodes
        added -= 2 if number_data > 0 else 0
        try:
            infos[i, 6, :, name_dict[Data[4 + i * 10 + added]]] = 1  # feature of trouble node number
            Y_in = np.sqrt(np.power(Y_process(Data[6 + i * 10 + added][0]), 2) +
                           np.power(Y_process(Data[6 + i * 10 + added][1]), 2))
            Y[:, 1, :, :] = Y_in  # the Y(in accident)
            Y_after = np.sqrt(np.power(Y_process(Data[6 + i * 10 + added][2]), 2) +
                              np.power(Y_process(Data[6 + i * 10 + added][3]), 2))
            Y[:, 2, :, :] = Y_after  # the Y(after accident)
            where_are_nan = np.isnan(Data[11 + i * 10 + added])
            Data[11 + i * 10 + added][where_are_nan] = 200
            # labels[i] = np.array(list(map(int, (abs(Data[11 + i * 10 + added]) - 180) > 0)))
            labels[i] = label_data(Data[11 + i * 10 + added])
            infos[i, :6, :, :] = np.transpose(Data[12 + i * 10 + added].reshape(33, num_nodes, 6), [2, 0, 1])
        except IndexError:
            print(i)
    return num_generator, Y, infos, labels


class trainSet(Dataset):
    def __init__(self, num_nodes, num, number_data):
        if type(number_data) is int:
            _, Y, infos, labels = read_data(num_nodes, num, number_data)
        if type(number_data) is list:
            if len(number_data) == 0:
                print('please input the data_number you want to use')
                raise
            else:
                num_generator, Y, infos, labels = read_data(num_nodes, num, number_data[0])
                for i in range(1, len(number_data)):
                    _, Y_, infos_, labels_ = read_data(num_nodes, num, number_data[i])
                    Y = np.concatenate((Y, Y_))
                    infos = np.concatenate((infos, infos_))
                    labels = np.concatenate((labels, labels_))

        self.Y = Y  # the matrix Y
        self.data = infos  # the time-dependent data
        self.result = labels  # the ground truth for nodes

    def __getitem__(self, index):
        return self.Y[index], self.data[index], self.result[index]

    def __len__(self):
        num, _ = self.result.shape
        return num


if __name__ == '__main__':
    a = 1
    print(a)
    exit(0)
    import networkx as nx

    a = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    net = dgl.from_networkx(nx.from_numpy_matrix(a))
    net.ndata['feats'] = torch.randn((3, 5))
    print(net.edges())
    a = net.nodes[0].data['feats']
    b = net.nodes[1].data['feats']
    print(a.shape)
    c = torch.cat([a, b], 1)
    print(c.shape)
    print(net.nodes[0].data['feats'].shape)
    exit(0)
    arr = [1, 2, 3, 4, 5]
    print(arr[-4:])
    exit(0)
