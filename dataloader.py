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
from pythonPS.sample_generator import RayWorkerForSampleGenerator


class trainSet(Dataset):
    def __init__(self, num, num_nodes, num_features, num_frames):
        self.Y = np.zeros([num, num_nodes, num_nodes, num_frames])          # the time-dependent matrix Y
        self.data = np.zeros([num, num_nodes, num_features, num_frames])    # the time-dependent data
        self.result = np.ones([num, num_nodes])                                        # the ground truth for nodes

    def __getitem__(self, index):
        return self.Y[index], self.data[index], self.result[index]

    def __len__(self):
        num, _ = self.result.shape
        return num


if __name__ == '__main__':
    train_data = trainSet(10, 5, 4, 33)
    trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
    for epoch in range(2):
        for i, datas in enumerate(trainloader):
            Y, data, result = datas
            print(Y.shape)



 
