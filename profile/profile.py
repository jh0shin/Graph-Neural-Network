#!/usr/bin/env python3
'''
Comparing computational time of
Pytorch Geometric vs Deep Graph Library
'''

# To training neural network with GPU, you have to remove (#) in
# CPU -> cuda memory copy line of graph and Net

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
# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
dataset = PygNodePropPredDataset(name=args.dataset, transform=T.ToSparseTensor()) 

split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].to(device)
test_idx = split_idx["test"].to(device)

data = dataset[0] # pyg graph object
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

model = PYG_Net(data.num_features, args.hidden_channel,
                dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
evaluator = Evaluator(name=args.dataset)

pyg_train(model, optimizer, data, train_idx)

acc = pyg_evaluate(model, evaluator, data, test_idx)
print('Accuracy: {:.4f}'.format(acc))