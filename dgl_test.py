#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : dgl_test.py    
@Modify Time      @Author    @Version    @Desciption
2020/8/9 15:17   TANG       1.0         test for the fundamental function of dgl
"""
import torch
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citation_graph
import dgl
import networkx as nx
import matplotlib.pyplot as plt

print(dgl.__version__)

def load_cora():
    data = citation_graph.load_cora()
    features = torch.from_numpy(data.features)
    labels = torch.from_numpy(data.labels)
    masks = torch.from_numpy(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, masks


if __name__ == '__main__':
    # g, features, labels, masks = load_cora()
    # print(features.type())
    g = dgl.DGLGraph()
    g.add_nodes(10)
    # 逐条往图中添加边
    for i in range(1, 4):
        g.add_edge(i, 0)
    # 批添加
    src = list(range(5, 8))
    dst = [0] * 3
    g.add_edges(src, dst)

    src = torch.tensor([8, 9])
    dst = torch.tensor([0, 0])
    g.add_edges(src, dst)

    plt.figure(figsize=(14, 6))
    nx.draw(g.to_networkx(), with_labels=True)
    plt.savefig("Graph.png", format="PNG")
