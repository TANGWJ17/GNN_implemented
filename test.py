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


def evaluate(model, data_iter, num_epoch):
    model.eval()
    loss_total = 0
    accu = 0
    with torch.no_grad():
        for epoch in num_epoch:
            try:
                num_generator, Y, infos, labels = next(data_iter)
                Y, infos, labels = Y.cuda(), infos.cuda(), labels.cuda()
            except StopIteration:
                batch_iterator = iter(data_iter)
                num_generator, Y, infos, labels = next(batch_iterator)
                Y, infos, labels = Y.cuda(), infos.cuda(), labels.cuda()
            label_predicted = model(Y, infos)
            loss_total += F.cross_entropy(label_predicted, labels)

 
