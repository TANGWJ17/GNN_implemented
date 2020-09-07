#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : loss.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 11:21   TANG       1.0         several loss function
"""
import warnings
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import torch.nn.functional as F
import seaborn as sns
import re


def MSE_loss(outputs, labels):
    """
    :param outputs: [batches, num_nodes]
    :param labels:  [batches, num_nodes]
    :return: float
    """
    if not (labels.size() == outputs.size()):
        warnings.warn("Using a ground truth size ({}) that is different to the prediction size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(labels.size(), outputs.size()),
                      stacklevel=2)
    loss = nn.MSELoss()
    return 0.5 * loss(outputs.float(), labels.float())


def Huber_loss(outputs, labels, delta=0.2):
    """
        :param outputs: [batches, num_nodes]
        :param labels:  [batches, num_nodes]
        :param delta:   float hyperparameter of function
        :return: float

        .. math::
            L_\delta(y,f(x)) = \left \{ \begin{array}{c} \frac{1}{2} (y - f(x))^2 & \mid y - f(x)
            \mid \leq \delta \\ \delta \mid y-f(x) \mid - \frac{1}{2} \delta ^2 & \text{otherwise}\end{array}\right
    """

    if not (labels.size() == outputs.size()):
        warnings.warn("Using a ground truth size ({}) that is different to the prediction size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(labels.size(), outputs.size()),
                      stacklevel=2)
    t = torch.abs(outputs - labels)
    loss = torch.mean(torch.where(t < delta, 0.5 * t ** 2, delta * (t - 0.5 * delta)))
    return loss


def draw_loss(loss_path):
    itera = []
    loss = []
    lines = open(loss_path, 'r').readlines()
    for line in lines:
        itera.append(int(line.strip().split('epoch:')[1].split('/')[0]))
        loss.append(float(line.strip().split('loss:')[1]))
    return itera, loss


def plot(matrix):
    sns.set()
    f, ax = plt.subplots()
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    matrix = np.around(matrix, decimals=2)
    # print(matrix)  # 打印出来看看
    sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    plt.xticks(np.array(range(2)), ['stable', 'unstable'])
    plt.yticks(np.array(range(2)), ['stable', 'unstable'])
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig('../data/confusion_matrix.png')

if __name__ == '__main__':
    file_path = '../logs/loss_accu.log'
    itera, loss = draw_loss(file_path)
    pl.plot(itera, loss, 'b-')
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')
    plt.title('Training loss with iteration')
    plt.savefig('../data/Loss.png')
