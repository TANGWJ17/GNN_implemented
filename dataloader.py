#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : dataloader.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/18 21:16   TANG       1.0         dataloader for networks
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

def read_data(num_nodes, num):
    all_Data = np.load('./data/sample.npz', allow_pickle=True)
    Data = all_Data['arr_0']
    name_dict = dict()
    node_names = Data[0]
    num_generator = 10
    # for node_name in node_names:
    #     num_generator += 1 if '发电' in node_name else 0
    for i in range(len(node_names)):
        name_dict[node_names[i]] = i
    Y = np.zeros([num, 3, num_nodes, num_nodes])   # num * 3(before/ing/after) * num_nodes * num_nodes
    infos = np.zeros([num, 6 + 1, 33, num_nodes])  # num * feature(+1) * frames * num_nodes
    labels = np.zeros([num, 3])  # num * num_generator
    Y_raw = np.sqrt(np.power(Data[1][0], 2) + np.power(Data[1][1], 2))
    Y[:, 0, :, :] = Y_raw # the original Y(before accident)
    for i in range(num_nodes):
        infos[i, 5, :, name_dict[data[4 + i * 10]]] = 1  # feature of trouble node number
        Y_in = np.sqrt(np.power(Data[6 + i * 10][0], 2) + np.power(Data[6 + i * 10][1], 2))
        Y[:, 1, :, :] = Y_in  # the Y(in accident)
        Y_after = np.sqrt(np.power(Data[6 + i * 10][2], 2) + np.power(Data[6 + i * 10][3], 2))
        Y[:, 2, :, :] = Y_after  # the Y(after accident)
        labels[i] = abs(data[11 + i * 10]) > 180
        infos[i, :6, :, : ] = data[12 + i * 10].reshape(33, 9, 6).transpose([1, 2, 0])
    for i in range(num_nodes, num):
        added = (i - num_nodes) // num_nodes
        try:
            infos[i, 5, :, name_dict[data[5 + i * 10 + added]]] = 1  # feature of trouble node number
            Y_in = np.sqrt(np.power(Data[7 + i * 10 + added][0], 2) + np.power(Data[7 + i * 10 + added][1], 2))
            Y[:, 1, :, :] = Y_in  # the Y(in accident)
            Y_after = np.sqrt(np.power(Data[7 + i * 10 + added][2], 2) + np.power(Data[7 + i * 10 + added][3], 2))
            Y[:, 2, :, :] = Y_after  # the Y(after accident)
            labels[i] = abs(data[12 + i * 10 + added]) > 180
            infos[i, :6, :, :] = data[13 + i * 10 + added].reshape(33, 9, 6).transpose([1, 2, 0])
        except IndexError:
            print(i)
    return num_generator, Y, infos, labels

class trainSet(Dataset):
    def __init__(self, num_nodes, num):
        num_generator, Y, infos, labels = read_data(num_nodes, num)
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
    a = np.array([-1919.2925, 59958.617,  -1885.8545, -1911.1536, -1931.381,  -1911.8141,
        -1929.3092 ,-1877.5665 ,-1904.0421 ,-2649.57])
    b = abs(a)
    print(b)
    c = bool(np.subtract(b, 180))
    print(c)
    exit(0)
    all_data = np.load('./data/sample_0_0.npz', allow_pickle=True)
    data = all_data['arr_0']
    print(data[0])
    for i in range(39):
        print('epoch: ' + str(i + 1))
        print(data[4 + i * 10])  # trouble node name
        print(data[9 + i * 10])  # stable or not
        print(data[11 + i * 10])  # power angle
        # print(data[12 + i * 10].shape)  # raw data without trouble node number
    for i in range(39, 8151):
        added = (i - 39) // 39
        # print('epoch: ' + str(i + 1))
        # print(data[5 + i * 10 + added])
        try:
            print(data[10 + i * 10 + added])
            print(data[12 + i * 10 + added])
            # a = abs(data[12 + i * 10 + added]) > 180
        except IndexError:
            print(i)
    # a, num = read_data()
    # print(a)
    # print(num)




 
