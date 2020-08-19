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
import torch.nn.functional as F

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



