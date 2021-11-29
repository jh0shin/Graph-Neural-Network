#!/usr/bin/env python3
'''
Comparing computational time of
Pytorch Geometric vs Deep Graph Library

Custom GraphConv2 kernel modified to profiling detail forward info
(Add) /home/{usr}/.local/lib/python3.8/site-packages/dgl/nn/pytorch/conv/graphconv_profile.py
(Modify) /home/junho/.local/lib/python3.8/site-packages/dgl/nn/pytorch/conv/__init__.py
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
import torch.nn.functional as F

from torch_sparse.tensor import SparseTensor

from torch_geometric.nn import Sequential, GCNConv
from torch.nn import ReLU, LogSoftmax

from dgl import from_scipy
from dgl.transform import add_self_loop
from dgl.nn.pytorch import Sequential as DglSequential
from dgl.nn import GraphConv

# random sparse matrix
import scipy.sparse as sparse
import scipy.stats as stats
import numpy as np

### timer
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from util.timer import Timer

import csv

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

# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Timer setting
timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

########################################
### Argparse
########################################
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channel', type=int, default=128)
parser.add_argument('-ms', type=int, default=128)
parser.add_argument('-d', type=float, default=0.01)
parser.add_argument('-nnf', type=int, default=128)
parser.add_argument('-nc', type=int, default=10)
parser.add_argument('-e', type=int, default=100)
args = parser.parse_args()

### Constants
EPOCH = args.e

MAT_SIZE = args.ms
DENSITY = args.d
RANDSEED = 34
NUM_NODE_FEATURES = args.nnf
NUM_CLASSES = args.nc

########################################
### Dataset
########################################
'''
    adj : scipy.sparse
        adjacent matrix of graph with MAT_SIZE nodes
        (MAT_SIZE, MAT_SIZE) coo sparse matrix
    spmat : torch.tensor
        node feature matrix
        (MAT_SIZE, NUM_NODE_FEATURES) dense matrix
    train_idx : torch.tensor
        train data index
        (MAT_SIZE * 0.8) dense matrix
    test_idx : torch.tensor
        test data index
        (MAT_SIZE * 0.2) dense matrix
'''
print(f"Number of node : {MAT_SIZE}, Number of node features : {NUM_NODE_FEATURES}, Number of classes : {NUM_CLASSES}")
print(f"Density : {DENSITY}, Random seed : {RANDSEED}, Epoch : {EPOCH}")
np.random.seed(RANDSEED)
gen = torch.manual_seed(RANDSEED)

with timer("Random data processing"):
    adj = sparse.random(MAT_SIZE, MAT_SIZE, density=DENSITY, dtype=np.float32, random_state=np.random.default_rng())
    node_features = torch.normal(mean=0, std=1, size=(MAT_SIZE, NUM_NODE_FEATURES), generator=gen).to('cuda')
    label = torch.randint(NUM_CLASSES, (MAT_SIZE, )).to('cuda')
    train_idx = torch.from_numpy(np.random.choice(np.arange(MAT_SIZE), size=int(MAT_SIZE*0.8), replace=False)).to('cuda')
    test_idx = torch.from_numpy(np.array(list(set(range(MAT_SIZE)) - set(train_idx.tolist())))).to('cuda')

########################################
### Pytorch Geometric
########################################
with timer("PYG"):
    # PyG data
    data_pyg = SparseTensor.from_scipy(adj).to('cuda')

    model = Sequential('x, edge_index', [
        (GCNConv(NUM_NODE_FEATURES, args.hidden_channel), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(args.hidden_channel, args.hidden_channel), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(args.hidden_channel, NUM_CLASSES), 'x, edge_index -> x'),
        LogSoftmax(dim=1),
    ]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(EPOCH):      
        optimizer.zero_grad()
        out = model(node_features, data_pyg)[train_idx]
        loss = F.nll_loss(out, label[train_idx])
        loss.backward()
        optimizer.step()

    # model.eval()
    # with torch.no_grad():
    #     out = model(node_features, data_pyg)[test_idx]
    #     y_pred = out.argmax(dim=-1, keepdim=True)
    #     correct = torch.sum(y_pred.squeeze(1) == label[test_idx])
    # print('Accuracy: {:.4f}'.format(correct.item() * 1.0 / len(y_pred)))

########################################
### Deep Graph Library
########################################
with timer("DGL"):
    # DGLGraph data
    data_dgl = from_scipy(adj).to('cuda')
    # _coo = adj.tocoo()
    # row, col, data = _coo.row, _coo.col, _coo.data
    # data_dgl = from_scipy(sparse.coo_matrix((data, (col, row))), eweight_name='w').to('cuda')
    data_dgl = add_self_loop(data_dgl)

    model = DglSequential(
        GraphConv(NUM_NODE_FEATURES, args.hidden_channel, activation=F.relu),
        GraphConv(args.hidden_channel, args.hidden_channel, activation=F.relu),
        GraphConv(args.hidden_channel, NUM_CLASSES, activation=F.log_softmax),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(EPOCH):
        out = model(data_dgl, node_features)
        loss = F.nll_loss(out[train_idx], label[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # model.eval()
    # with torch.no_grad():
    #     logits = model(data_dgl, node_features)
    #     _, indices = torch.max(logits[test_idx], dim=1)
    #     correct = torch.sum(indices == label[test_idx])
    # print('Accuracy: {:.4f}'.format(correct.item() * 1.0 / len(indices)))

# print(timer.summary())

result = timer.return_summary()
f = open('crosskernel/benchmark2.csv', 'a', newline='')
wr = csv.writer(f)
wr.writerow([
    MAT_SIZE, DENSITY, EPOCH,
    result['DGL']['total_time'], result['PYG']['total_time']
])
# print([
#     MAT_SIZE, DENSITY, EPOCH,
#     result['DGL']['total_time'], result['PYG']['total_time']
# ])