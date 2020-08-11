#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : dgl_test.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/9 15:17   TANG       1.0         test for the fundamental function of dgl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools


# 定义 message function and reduce function（重要）
def gcn_message(edges):
    # 该函数批量处理边
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg': edges.src['h']}  # edges.src.data指的是获取边出发节点的‘h’属性信息

def gcn_reduce(nodes):
    # 该函数批量处理节点
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    # 表示将附近有连接的节点的‘msg’属性数据按照'dim=1'进行加和，成为该节点新的‘h’属性数据
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# 定义GCNLayer
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, g, inputs):
        # g 是图（graph） 并且 inputs 是输入的节点特征向量
        # 首先设置图中 节点的特征向量
        g.ndata['h'] = inputs
        # 触发在所有边上传递信息
        g.send(g.edges(), gcn_message)
        # 触发在所有边上聚集信息
        g.recv(g.nodes(), gcn_reduce)
        # 提取边缘特征的结果
        h = g.ndata.pop('h')
        # 进行线性变换
        return self.linear(h)

# 定义GCN，由两层GCNLayer构成
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, num_classes)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

def build_karate_club_graph():
    # 所有78条边都存储在两个numpy数组中。一个用于源端点
    # 另一个用于目标端点。
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # 在DGL中，边是有方向性的;让他们双向。
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # 构造一个DGLGraph
    return dgl.DGLGraph((u, v))

if __name__ == '__main__':
    G = build_karate_club_graph()
    # 图的可视化，由于实际图形是无向的，因此我们去掉边的方向，以达到可视化的目的
    # nx_G = G.to_networkx().to_undirected()
    # # 为了图更加美观，我们使用Kamada-Kawaii layout
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.savefig('graph.png', format='PNG')
    net = GCN(5, 5, 2)  # 输出特征为2，可以用于可视化
    embed = nn.Embedding(34, 5)  # 34为节点数，5为特征维度
    inputs = embed.weight
    # 设置半监督训练，0和33号节点给定label
    labeled_nodes = torch.tensor([0, 33])  # 只有教练（节点0）和俱乐部主席（节点33）是被标记
    labels = torch.tensor([0, 1])  # 他们的标签是不同的
    # start training
    optim = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
    all_logits = []
    for epoch in range(0, 50):
        logits = net(G, inputs)  # net() 是一个GCN模型 G是一个图数据 inputs 是节点 embedding
        all_logits.append(logits.detach())  # 保存所有图，用于后续可视化
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logp[labeled_nodes], labels)  # 仅计算有标签的数据
        optim.zero_grad()  # 清零梯度
        loss.backward()    # 反向传播
        optim.step()       # 参数更新
        print('epoch:{} | loss:{:.4f}'.format(epoch, loss.item()))
