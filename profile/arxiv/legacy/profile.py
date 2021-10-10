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
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T


### GCN network declaration
from gcn import *

### Dataset
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset

### timer
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from util.timer import Timer

########################################
### Argparse
########################################
parser = argparse.ArgumentParser()
parser.add_argument('--platform', type=str)
parser.add_argument('--dataset', type=str, default='obgn-arxiv')
parser.add_argument('--hidden_channel', type=int, default=128)
args = parser.parse_args()

########################################
### Pytorch Geometric - GCN
########################################
if args.platform == 'pyg':
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Timer setting
    timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

    # Dataset
    dataset = PygNodePropPredDataset(name=args.dataset, transform=T.ToSparseTensor()) 

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    test_idx = split_idx["test"]

    data = dataset[0] # pyg graph object
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    model = PYG_Net(data.num_features, args.hidden_channel,
                    dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    evaluator = Evaluator(name=args.dataset)

    for _ in range(_EPOCH):
        pyg_train(model, optimizer, data, train_idx, timer)

    # acc = pyg_evaluate(model, evaluator, data, test_idx)
    # print('Accuracy: {:.4f}'.format(acc))

    print(timer.summary())

########################################
### Deep Graph Library - GCN
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

    model = DGL_Net(len(features[0]), args.hidden_channel,
                    dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    evaluator = Evaluator(name=args.dataset)

    for _ in range(_EPOCH):
        dgl_train(model, optimizer, data, features, label, train_idx, timer)

    # acc = dgl_evaluate(model, data, features, label, test_idx)
    # print('Accuracy: {:.4f}'.format(acc))

    print(timer.summary())