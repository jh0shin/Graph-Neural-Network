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
from torch_sparse import SparseTensor

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
parser.add_argument('--cluster_number', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

########################################
### Pytorch Geometric - GCN
########################################
# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Timer
timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

# Dataset
dataset = PygNodePropPredDataset(name=args.dataset) 

graph = dataset[0]

adj = SparseTensor(row=graph.edge_index[0],
                    col=graph.edge_index[1])

split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].tolist()
test_idx = split_idx["test"].tolist()

model = PYG_Net(graph.num_features, args.hidden_channel,
                dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
evaluator = Evaluator(name=args.dataset)

for _ in range(_EPOCH):
    with timer("batch generate"):
        # generate batches
        parts = random_partition_graph(graph.num_nodes,
                                        cluster_number=args.cluster_number)
        data = generate_sub_graphs(adj, parts, cluster_number=args.cluster_number,
                                    batch_size=args.batch_size)

    pyg_train(model, optimizer, graph, data, train_idx, device, timer)

# acc = pyg_evaluate(model, evaluator, graph, adj, test_idx)
# print('Accuracy: {:.4f}'.format(acc))

print(timer.summary())
sys.exit()

########################################
### Deep Graph Library - GCN
########################################
# Dataset
dataset = DglNodePropPredDataset(name=args.dataset, transform=T.ToSparseTensor()) 

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

for _ in range(_EPOCH):
    dgl_train(model, optimizer, data, train_idx)

acc = dgl_evaluate(model, evaluator, data, test_idx)
print('Accuracy: {:.4f}'.format(acc))