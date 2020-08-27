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
from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import trainSet
from baseline import Baseline
from gat import GAT, GAT_edge
from loss import MSE_loss, Huber_loss
import argparse

parser = argparse.ArgumentParser(
        description='GNN used in fault nodes prediction')
parser.add_argument('--model', default='baseline', type=str, choices=['baseline', 'GAT', 'GAT_edge'],
                        help='choose the model you need to train')
parser.add_argument('--batch_size', default=4, type=int,
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
    if args.model is 'baseline':
        net = Baseline()
    elif args.model is 'GAT':
        net = GAT()
    elif args.model is 'GAT_edge':
        net = GAT_edge()
    else:
        print('must choose a model in the choices')
        raise

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()


    data_amount = 8151
    train_data = trainSet(39, 8151)
    trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
    batch_loader = iter(trainloader)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    net.train()
    for epoch in range(data_amount // args.batch_size):
        # load train data
        try:
            Y, infos, labels = next(batch_loader)
        except StopIteration:
            batch_iterator = iter(trainloader)
            Y, infos, labels = next(batch_iterator)

        label_predicted = net(Y, infos)
        loss = Huber_loss(label_predicted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            torch.save(net.state_dict(),
                       args.save_folder + '/' + args.model_name +
                       '_' + str(epoch) + '_' + str(loss) + '.pth')
        print('epoch:{} | loss:{:.4f}'.format(epoch, loss.item()))
    torch.save(net.state_dict(),
               args.save_folder + '/' + args.model_name + '.pth')

if __name__ == '__main__':
    train()
