#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : train.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 9:37   TANG       1.0         script for training model
"""
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(
        description='GNN used in fault nodes prediction')
parser.add_argument('--model', default='baseline', type=str, choices=['baseline', 'GAT', 'GAT_edge'],
                        help='choose the model you need to train')
parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
parser.add_argument('--cuda', default=False, type=bool,
                        help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        help='initial learning rate')
parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
parser.add_argument('--log_folder', default='logs/',
                        help='Directory for saving logging')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    print(args)

if __name__ == '__main__':
    train()
