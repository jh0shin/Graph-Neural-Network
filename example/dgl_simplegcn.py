#!/usr/bin/env python3
###
# test code from https://towardsdatascience.com/start-with-graph-convolutional-neural-networks-using-dgl-cf9becc570e1
###

# To training neural network with GPU, you have to remove (#) in
# CPU -> cuda memory copy line of graph and Net

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv

import numpy as np

########################################
# Import Cora dataset
########################################
dataset = CoraGraphDataset()
g = dataset[0]
g = g.to('cuda:0')                # graph to cuda memory
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
test_mask = g.ndata['test_mask']

########################################
# GCN network class declaration
########################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GraphConv(1433, 8*16) #activation default=None
        self.layer2 = GraphConv(8*16, 7)    #activation default=None


    def forward(self, g, features):
        x1 = F.relu(self.layer1(g, features)) #ReLU activation function
        x2 = self.layer2(g, x1)
        return x2

net = Net()
net = net.cuda()                  # GCN to cuda memory

########################################
# Evaluate function declaration
########################################
def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

########################################
# Training
########################################
g.add_edges(g.nodes(), g.nodes())
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
loss_list=[]
acc_list=[]
all_logits=[]
for epoch in range(1):
    net.train()
    logits = net(g, features)
    
    #print(logits)
    logp = F.log_softmax(logits, 1)
    all_logits.append(logp)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = evaluate(net, g, features, labels, test_mask)
    loss_list.append(loss.item())
    acc_list.append(acc)
    print(f"{epoch} epoch - accuracy : {acc}  loss : {loss.item()}")