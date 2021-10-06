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
### Common library
import torch
import torch.nn as nn
import torch.nn.functional as F

### Pytorch Geometric
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

### Deep Graph Library
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv

### Pytorch profiler
import torch.autograd.profiler as profiler

########################################
### GCN network class declaration
########################################
class DGL_Net(nn.Module):
    def __init__(self):
        super(DGL_Net, self).__init__()
        self.layer1 = GraphConv(1433, 256)
        self.layer2 = GraphConv(256, 32)
        self.layer3 = GraphConv(32, 7)

    def forward(self, g, features):
        with profiler.record_function("FORWARD"):
            x1 = F.relu(self.layer1(g, features))
            x2 = F.relu(self.layer2(g, x1))
            x3 = self.layer3(g, x2)
            x4 = F.log_softmax(x3, dim=1)
        return x4

class PYG_Net(torch.nn.Module):
    def __init__(self):
        super(PYG_Net, self).__init__()
        self.layer1 = GCNConv(pyg_dataset.num_node_features, 256)
        self.layer2 = GCNConv(256, 32)
        self.layer3 = GCNConv(32, pyg_dataset.num_classes)

    def forward(self, data):
        with profiler.record_function("FORWARD"):
            g, edge_index = data.x, data.edge_index

            x1 = F.relu(self.layer1(g, edge_index))
            x2 = F.relu(self.layer2(x1, edge_index))
            x3 = self.layer3(x2, edge_index)
            x4 = F.log_softmax(x3, dim=1)
        return x4

########################################
### Evaluate function declaration
########################################
def dgl_evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def pyg_evaluate(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    return correct / data.test_mask.sum().item()

########################################
### Training function declaration
########################################
def dgl_train(model, optimizer, g, features, labels, mask):
    model.train()
    for epoch in range(_EPOCH):
        if epoch == _EPOCH - 1:
            with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
                out = model(g, features)
            print(prof.key_averages().table(sort_by="cuda_time_total"))
        else:
            out = model(g, features)

        all_logits.append(out)
        loss = F.nll_loss(out[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def pyg_train(model, optimizer, data):
    model.train()
    for epoch in range(_EPOCH):
        optimizer.zero_grad()

        if epoch == _EPOCH - 1:
            with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
                out = out = model(data)
            print(prof.key_averages().table(sort_by="cuda_time_total"))
        else:
            out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    

########################################
### Import Cora dataset
########################################
### device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### DGL dataset
dgl_dataset = CoraGraphDataset()
g = dgl_dataset[0].to(device)
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
test_mask = g.ndata['test_mask']

### PYG dataset
pyg_dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = pyg_dataset[0].to(device)

### network
warmup = DGL_Net().to(device)
dgl_net = DGL_Net().to(device)
pyg_net = PYG_Net().to(device)

### optimizer
dgl_optimizer = torch.optim.Adam(dgl_net.parameters(), lr=0.01)
pyg_optimizer = torch.optim.Adam(pyg_net.parameters(), lr=0.01)

########################################
### Training
########################################
### DGL
g.add_edges(g.nodes(), g.nodes())
all_logits=[]

# ### warming up
# dgl_train(warmup, dgl_optimizer, g, features, labels, train_mask)
# all_logits=[]

# with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
#     dgl_train(dgl_net, dgl_optimizer, g, features, labels, train_mask)
# print(prof.key_averages().table(sort_by="cuda_time_total"))

dgl_train(dgl_net, dgl_optimizer, g, features, labels, train_mask)

### PYG
# with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
#     pyg_train(pyg_net, pyg_optimizer, data)
# print(prof.key_averages().table(sort_by="cuda_time_total"))

pyg_train(pyg_net, pyg_optimizer, data)

########################################
### Result
########################################
### Accurary
acc = dgl_evaluate(dgl_net, g, features, labels, test_mask)
print('Accuracy: {:.4f}'.format(acc))

acc = pyg_evaluate(pyg_net, data)
print('Accuracy: {:.4f}'.format(acc))