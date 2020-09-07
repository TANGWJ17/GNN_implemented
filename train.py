#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : train.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/19 9:37   TANG       1.0         script for training model
"""
import os
import sys
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import init
from dataloader import trainSet
from baseline import Baseline
from test import evaluate
from gat import GAT, GAT_edge
from loss import MSE_loss, Huber_loss
import argparse

torch.manual_seed(1)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://101.6.64.59:8002', rank=0, world_size=1)
parser = argparse.ArgumentParser(
        description='GNN used in fault nodes prediction')
parser.add_argument('--model', default='baseline', type=str, choices=['baseline', 'GAT', 'GAT_edge'],
                        help='choose the model you need to train')
parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
parser.add_argument('--cuda', default=True, type=bool,
                        help='using cuda for accelerating')
parser.add_argument('--init_type', default='xavier', type=str, choices=['normal', 'xavier', 'kaiming', 'orthogonal'],
                        help='several init method')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)

def train():
    if args.model is 'baseline':
        net = Baseline(in_channels=7, out_channels_1=7, out_channels_2=7, KT_1=4, KT_2=3, num_nodes=39,
                       batch_size=args.batch_size, frames=33, frames_0=12, num_generator=10)
    elif args.model is 'GAT':
        net = GAT()
    elif args.model is 'GAT_edge':
        net = GAT_edge()
    else:
        print('must choose a model in the choices')
        raise

    if args.init_type is not None:
        try:
            init_weights(net, init_type=args.init_type)
        except:
            sys.exit('Load Network  <==> Init_weights error!')

    # net = nn.DataParallel(net)
    net = net.cuda()

    accuracy = 0
    train_file = 4
    train_amount = 6400 # 8144
    eval_amount = 3200
    num_epoch = train_amount // args.batch_size * train_file
    train_data = trainSet(39, train_amount, [0, 1, 2, 3])
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    batch_loader = iter(trainloader)
    eval_data = trainSet(39, eval_amount, 4)
    evalloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=True)
    eval_iter = iter(evalloader)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    net.train()
    #  train ------------------------------------------------
    print('---- epoch start ----')
    start_time = time.time()
    for epoch in range(num_epoch):
        # load train data
        try:
            Y, infos, labels = next(batch_loader)
            Y, infos, labels = Y.float().cuda(), infos.float().cuda(), labels.float().cuda()
        except StopIteration:
            batch_iterator = iter(trainloader)
            Y, infos, labels = next(batch_iterator)
            Y, infos, labels = Y.float().cuda(), infos.float().cuda(), labels.float().cuda()
        label_predicted = net(Y, infos)
        loss = MSE_loss(label_predicted, labels.long())
        # loss = F.cross_entropy(label_predicted, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:{}/{} | loss:{:.4f}'.format(epoch + 1, num_epoch, loss.item()))
        with open(args.log_folder + 'loss.log', mode='a') as f:
            f.writelines('\n epoch:{}/{} | loss:{:.4f}'.format(epoch + 1, num_epoch, loss.item()))

        #  eval ------------------------------------------------
        if epoch % 20 == 0:
            net.eval()
            accu, _ = evaluate(model=net, data_iter=eval_iter, data_loader=evalloader, num_epoch=10)
            print('accuracy:{}'.format(accu))
            with open(args.log_folder + 'accu.log', mode='a') as f:
                f.writelines('\n eval epoch:{} | loss:{:.4f}'.format(epoch // 20 + 1, loss.item()))
            if accu > accuracy:
                torch.save(net.state_dict(), args.save_folder + '{}_{}.pth'.format(args.model, accu))
                accuracy = accu

    stop_time = time.time()
    print("program run for {} s".format(stop_time - start_time))

if __name__ == '__main__':
    train()
