#!/usr/bin/env python3
'''
GCN network and function declaration
'''

### Constants
_EPOCH = 100

########################################
### Library
########################################
### Common library
import torch
import torch.nn as nn
import torch.nn.functional as F

### Pytorch Geometric
from torch_geometric.nn import GCNConv

### Deep Graph Library
from dgl.nn import GraphConv

### Evaluator
from ogb.nodeproppred import Evaluator

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
    for epoch in range(_EPOCH):
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

def pyg_evaluate(model, evaluator, data, test_idx):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    return evaluator.eval({
        'y_true': data.y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

def pyg_train(model, optimizer, data, train_idx):
    model.train()

    for epoch in range(_EPOCH):
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()