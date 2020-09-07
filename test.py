#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : test.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 10:24   TANG       1.0         None
"""
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import trainSet
from baseline import Baseline
from gat import GAT, GAT_edge
from loss import MSE_loss, Huber_loss
from loss import plot
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate(model, data_iter, data_loader, num_epoch, threshold=0.5):
    model.eval()
    loss_total = 0
    accu = 0
    with torch.no_grad():
        for epoch in range(num_epoch):
            try:
                Y, infos, labels = next(data_iter)
                Y, infos, labels = Y.float().cuda(), infos.float().cuda(), labels.type(torch.int32).cuda()
            except StopIteration:
                batch_iterator = iter(data_loader)
                Y, infos, labels = next(batch_iterator)
                Y, infos, labels = Y.float().cuda(), infos.float().cuda(), labels.type(torch.int32).cuda()
            label_predicted = model(Y, infos)
            labels_threshold = label_predicted > threshold
            all_right = 1 - torch.mean((labels ^ labels_threshold).type(torch.float32))
            loss_total += Huber_loss(label_predicted, labels.long())
            accu += all_right

    accu /= num_epoch
    return accu, loss_total

@torch.no_grad()
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
    model.load_state_dict(torch.load(weigths_path))
    # model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    accu = 0
    true_labels = np.array([])
    pred_labels = np.array([])
    for epoch in range(num_epoch):
        try:
            Y, infos, labels = next(test_iter)
            Y, infos, labels = Y.float().cuda(), infos.float().cuda(), labels.type(torch.int32).cuda()
        except StopIteration:
            batch_iterator = iter(test_loader)
            Y, infos, labels = next(batch_iterator)
            Y, infos, labels = Y.float().cuda(), infos.float().cuda(), labels.type(torch.int32).cuda()
        label_predicted = model(Y, infos)
        labels_threshold = label_predicted > threshold
        true_labels = np.concatenate((true_labels, labels.reshape((1, -1)[0])))
        pred_labels = np.concatenate((pred_labels, labels_threshold.reshape((1, -1)[0])))
        all_right = 1 - torch.mean((labels ^ labels_threshold).type(torch.float32))
        print('epoch:{}, accu:{}'.format(epoch, all_right))
        accu += all_right
    accu /= num_epoch
    plot(confusion_matrix(true_labels, pred_labels))
    return accu

if __name__ == '__main__':
    true_ = np.array([[0, 1, 1, 0], [1, 1, 1, 1]]).reshape((1, -1))
    print(true_[0])
    exit(0)
    pred_ = np.array([[1, 1, 1, 0], [1, 1, 1, 1]]).reshape((1, -1))
    print(true_[0])
    cn = confusion_matrix(true_[0], pred_[0])
    print(cn)
    exit(0)
    test_data = trainSet(39, 1600, 2)
    testloader = DataLoader(test_data, batch_size=16, shuffle=True)
    test_iter = iter(testloader)
    accu = test(test_iter, testloader, './weights/baseline_0.995.pth', 100, 0)
    print('acuu:{}'.format(accu))

 
