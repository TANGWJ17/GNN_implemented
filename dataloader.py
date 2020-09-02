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
from torch.utils.data import Dataset, DataLoader

def Y_process(Y_):
    row, col = np.diag_indices(Y_.shape[0])
    Y_[row, col] = np.zeros(Y_.shape[0])
    return Y_

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
    Y = np.zeros([num, 3, num_nodes, num_nodes])   # num * 3(before/ing/after) * num_nodes * num_nodes
    infos = np.zeros([num, 6 + 1, 33, num_nodes])  # num * feature(+1) * frames * num_nodes
    labels = np.zeros([num, num_generator])  # num * num_generator
    Y_raw = np.sqrt(np.power(Y_process(Y_RAW[0]), 2) + np.power(Y_process(Y_RAW[1]), 2))
    Y[:, 0, :, :] = Y_raw # the original Y(before accident)
    for i in range(num_nodes):
        added = -2 if number_data > 0 else 0
        infos[i, 6, :, name_dict[Data[4 + i * 10 + added]]] = 1  # feature of trouble node number
        Y_in = np.sqrt(np.power(Y_process(Data[6 + i * 10 + added][0]), 2) + np.power(Y_process(Data[6 + i * 10 + added][1]), 2))
        Y[:, 1, :, :] = Y_in  # the Y(in accident)
        Y_after = np.sqrt(np.power(Y_process(Data[6 + i * 10 + added][2]), 2) + np.power(Y_process(Data[6 + i * 10 + added][3]), 2))
        Y[:, 2, :, :] = Y_after  # the Y(after accident)
        where_are_nan = np.isnan(Data[11 + i * 10 + added])
        Data[11 + i * 10 + added][where_are_nan] = 200
        labels[i] = np.array(list(map(int, (abs(Data[11 + i * 10 + added]) - 180) > 0)))
        infos[i, :6, :, : ] = np.transpose(Data[12 + i * 10 + added].reshape(33, num_nodes, 6), [2, 0, 1])
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
            labels[i] = np.array(list(map(int, (abs(Data[11 + i * 10 + added]) - 180) > 0)))
            infos[i, :6, :, :] = np.transpose(Data[12 + i * 10 + added].reshape(33, num_nodes, 6), [2, 0, 1])
        except IndexError:
            print(i)
    return num_generator, Y, infos, labels

class trainSet(Dataset):
    def __init__(self, num_nodes, num, number_data):
        num_generator, Y, infos, labels = read_data(num_nodes, num, number_data)
        self.Y = Y            # the matrix Y
        self.data = infos     # the time-dependent data
        self.result = labels  # the ground truth for nodes

    def __getitem__(self, index):
        return self.Y[index], self.data[index], self.result[index]

    def __len__(self):
        num, _ = self.result.shape
        return num


if __name__ == '__main__':
    # train_data = trainSet(10, 5, 4, 33)
    # trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
    # for epoch in range(2):
    #     for i, datas in enumerate(trainloader):
    #         Y, data, result = datas
    #         print(Y.shape)


    # with open('label.txt', 'w') as f:
    #     for label in labels:
    #         f.writelines(str(label))
    # exit(0)
    # all_Data = np.load('./data/sample_0_{}.npz'.format(0), allow_pickle=True)
    # data = all_Data['arr_0']
    # np.savez('./data/Y_raw.npz', data[1])
    arr = [1, 2, 3, 4, 5]
    print(arr[-4:])
    exit(0)
    all_data = np.load('./data/Y_raw.npz', allow_pickle=True)
    data = all_data['arr_0']
    print(data.shape)
    exit(0)
    print(data[0])
    print(data[1])
    print(data[2])
    print(data[3])
    print(data[4])
    exit(0)
    for i in range(39):
        print('epoch: ' + str(i + 1))
        print(data[4 + i * 10 ])  # trouble node name
        print(data[9 + i * 10])  # stable or not
        print(data[11 + i * 10])  # power angle
        # print(data[12 + i * 10].shape)  # raw data without trouble node number
    exit(0)
    for i in range(39, 50):
        added = i // 39
        # print('epoch: ' + str(i + 1))
        print(data[4 + i * 10 + added - 2])
        try:
            print(data[9 + i * 10 + added - 2])
            print(data[11 + i * 10 + added - 2])
            # a = abs(data[12 + i * 10 + added]) > 180
        except IndexError:
            print(i)
    # a, num = read_data()
    # print(a)
    # print(num)




 
