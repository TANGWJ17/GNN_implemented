#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : test.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 10:24   TANG       1.0         None
"""
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import trainSet
from baseline import Baseline
from gat import GAT, GAT_edge
from loss import MSE_loss, Huber_loss
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def evaluate(model, data_iter, data_loader, num_epoch, threshold=0.5):
    model.eval()
    loss_total = 0
    accu = 0
    with torch.no_grad():
        for epoch in num_epoch:
            try:
                num_generator, Y, infos, labels = next(data_iter)
                Y, infos, labels = Y.cuda(), infos.cuda(), labels.cuda()
            except StopIteration:
                batch_iterator = iter(data_loader)
                num_generator, Y, infos, labels = next(batch_iterator)
                Y, infos, labels = Y.cuda(), infos.cuda(), labels.cuda()
            label_predicted = model(Y, infos)
            labels_threshold = label_predicted > threshold
            all_right = torch.mean(1 - labels ^ labels_threshold)
            loss_total += F.cross_entropy(label_predicted, labels)
            accu += all_right

    accu /= num_epoch
    return accu, loss_total

def test(test_iter, test_loader, weigths_path, num_epoch, model_type=0, threshold=0.5):
    if model_type == 0:
        model = Baseline(in_channels=7, out_channels_1=7, out_channels_2=7, KT_1=4, KT_2=3, num_nodes=39,
                       batch_size=16, frames=33, frames_0=12, num_generator=10)
    elif model_type == 1:
        model = GAT()
    elif model_type == 2:
        model = GAT_edge()
    else:
        raise
    model.load_state_dict(weigths_path)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    accu = 0
    with torch.no_grad:
        for epoch in num_epoch:
            try:
                num_generator, Y, infos, labels = next(test_iter)
                Y, infos, labels = Y.cuda(), infos.cuda(), labels.cuda()
            except StopIteration:
                batch_iterator = iter(test_loader)
                num_generator, Y, infos, labels = next(batch_iterator)
                Y, infos, labels = Y.cuda(), infos.cuda(), labels.cuda()
            label_predicted = model(Y, infos)
            labels_threshold = label_predicted > threshold
            all_right = torch.mean(1 - labels ^ labels_threshold)
            accu += all_right
    accu /= num_epoch
    return accu

if __name__ == '__main__':
    test_data = trainSet(39, 1600, 1)
    testloader = DataLoader(test_data, batch_size=16, shuffle=True)
    test_iter = iter(testloader)
    test(test_iter, testloader, )

 
