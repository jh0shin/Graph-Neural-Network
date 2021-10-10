#!/usr/bin/env python3
'''
Comparing computational time of
Pytorch Geometric vs Deep Graph Library
'''

# To training neural network with GPU, you have to remove (#) in
# CPU -> cuda memory copy line of graph and Net

### Constants
_EPOCH = 100

########################################
### Library
########################################
### Argparse
import argparse

### Common library
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import Sequential, GCNConv
from torch.nn import ReLU, LogSoftmax

from dgl.nn.pytorch import Sequential as DglSequential
from dgl.nn import GraphConv

### Dataset
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from ogb.nodeproppred import Evaluator

### timer
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from util.timer import Timer

########################################
### Utils
########################################
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
### Argparse
########################################
parser = argparse.ArgumentParser()
parser.add_argument('--platform', type=str)
parser.add_argument('--dataset', type=str, default='obgn-arxiv')
parser.add_argument('--hidden_channel', type=int, default=128)
args = parser.parse_args()

########################################
### Pytorch Geometric
########################################
if args.platform == 'pyg':
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Timer setting
    timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

    # Dataset
    dataset = PygNodePropPredDataset(name=args.dataset, transform=T.ToSparseTensor()) 

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)
    test_idx = split_idx["test"]

    graph = dataset[0]
    adj = graph.adj_t.to_symmetric()
    model = Sequential('x, edge_index', [
        (GCNConv(graph.num_features, args.hidden_channel), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(args.hidden_channel, args.hidden_channel), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(args.hidden_channel, dataset.num_classes), 'x, edge_index -> x'),
        LogSoftmax(dim=1),
    ]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    evaluator = Evaluator(name=args.dataset)

    data = dataset[0] # pyg graph object
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    model.train()
    for _ in range(_EPOCH):      
        optimizer.zero_grad()
        with timer("forward"): out = model(data.x, data.adj_t)[train_idx]
        with timer("loss"): loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        with timer("backward"): loss.backward()
        with timer("optimizer.step"): optimizer.step()

    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    print('Accuracy: {:.4f}'.format(evaluator.eval({
        'y_true': data.y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']))

    print(timer.summary())

########################################
### Deep Graph Library
########################################
if args.platform == 'dgl':
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = DglNodePropPredDataset(name=args.dataset) 

    # Timer setting
    timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)
    test_idx = split_idx["test"].to(device)

    data, label = dataset[0] # dgl graph object
    data.add_edges(data.nodes(), data.nodes())
    data = data.to(device)
    label = label.to(device)
    features = data.ndata['feat']
    # print(len(features), data.number_of_nodes(), len(features[0]))

    model = DglSequential(
        GraphConv(len(features[0]), args.hidden_channel, activation=F.relu),
        GraphConv(args.hidden_channel, args.hidden_channel, activation=F.relu),
        GraphConv(args.hidden_channel, dataset.num_classes, activation=F.log_softmax),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    evaluator = Evaluator(name=args.dataset)

    model.train()
    for _ in range(_EPOCH):
        with timer("forward"): out = model(data, features)
        with timer("loss"): loss = F.nll_loss(out[train_idx], label.squeeze(1)[train_idx])

        optimizer.zero_grad()
        with timer("backward"): loss.backward()
        with timer("optimizer.step"): optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data, features)
        _, indices = torch.max(logits[train_idx], dim=1)
        correct = torch.sum(indices == label.squeeze(1)[train_idx])
    print('Accuracy: {:.4f}'.format(correct.item() * 1.0 / len(indices)))

    print(timer.summary())