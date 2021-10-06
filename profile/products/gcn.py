#!/usr/bin/env python3
'''
GCN network and function declaration
'''

########################################
### Library
########################################
### Common library
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

### Pytorch Geometric
from torch_geometric.nn import GCNConv
import torch_geometric as tg

### Deep Graph Library
from dgl.nn import GraphConv

### Evaluator
from ogb.nodeproppred import Evaluator

########################################
### Utils
########################################
# Codes from : https://github.com/lightaime/deep_gcns_torch/blob/7885181484978fbf3839bf0e929fb1c2484d0a7d/utils/data_util.py#L14
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

# random partition graph
def random_partition_graph(num_nodes, cluster_number=10):
    parts = np.random.randint(cluster_number, size=num_nodes)
    return parts


def generate_sub_graphs(adj, parts, cluster_number=10, batch_size=1):
    # convert sparse tensor to scipy csr
    adj = adj.to_scipy(layout='csr')

    num_batches = cluster_number // batch_size

    sg_nodes = [[] for _ in range(num_batches)]
    sg_edges = [[] for _ in range(num_batches)]

    for cluster in range(num_batches):
        sg_nodes[cluster] = np.where(parts == cluster)[0]
        sg_edges[cluster] = tg.utils.from_scipy_sparse_matrix(adj[sg_nodes[cluster], :][:, sg_nodes[cluster]])[0]

    return sg_nodes, sg_edges

def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    # print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))

########################################
### Deep Graph Library - GCN
########################################
class DGL_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DGL_Net, self).__init__()
        self.layer1 = GraphConv(in_channels, hidden_channels)
        self.layer2 = GraphConv(hidden_channels, hidden_channels)
        self.layer3 = GraphConv(hidden_channels, out_channels)

    def forward(self, g, features):
        x1 = F.relu(self.layer1(g, features))
        x2 = F.relu(self.layer2(g, x1))
        x3 = self.layer3(g, x2)
        return F.log_softmax(x3, dim=1)

def dgl_evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def dgl_train(model, optimizer, g, features, labels, mask, all_logits):
    model.train()

    out = model(g, features)
    all_logits.append(out)
    loss = F.nll_loss(out[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

########################################
### Pytorch Geometric - GCN
########################################
class PYG_Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PYG_Net, self).__init__()
        self.layer1 = GCNConv(in_channels, hidden_channels)
        self.layer2 = GCNConv(hidden_channels, hidden_channels)
        self.layer3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, adj_t):
        x1 = F.relu(self.layer1(x, adj_t))
        x2 = F.relu(self.layer2(x1, adj_t))
        x3 = self.layer3(x2, adj_t)
        return F.log_softmax(x3, dim=1)

def pyg_evaluate(model, evaluator, data, adj, test_idx):
    model.eval()
    model.to('cpu')

    out = model(data.x, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)

    return evaluator.eval({
        'y_true': data.y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

def pyg_train(model, optimizer, graph, data, train_idx, device, timer):
    model.train()

    optimizer.zero_grad()

    with timer("train preprocess"):
        nodes, edges = data
        train_y = graph.y[train_idx].squeeze(1)

        idx_clusters = np.arange(len(nodes))
        np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x_ = graph.x[nodes[idx]].to(device)
        edges_ = edges[idx].to(device)
        mapper = {node: idx for idx, node in enumerate(nodes[idx])}

        inter_idx = intersection(nodes[idx], train_idx)
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        with timer("forward"): out = model(x_, edges_)
        target = train_y[inter_idx].to(device)

        with timer("loss"): loss = F.nll_loss(out[training_idx], target)
        with timer("backward"): loss.backward()
        with timer("optimizer.step"): optimizer.step()