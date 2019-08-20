#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/8 13:21

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
from model_component.conv.gat_conv_gn import GatConv
from model_component.utils.add_self_loop import add_self_loops
from data_loader import Cora
import torch.nn.functional as F

class GatNet(torch.nn.Module):
    def __init__(self, node_feature_size, label_feature_size):
        super(GatNet, self).__init__()
        self.p = 0.6
        self.size_first_layer = 8
        self.first_heads = 8

        self.conv1 = GatConv(node_feature_size, self.size_first_layer, heads=self.first_heads, dropout=self.p)
        self.conv2 = GatConv(self.size_first_layer * self.first_heads, label_feature_size, heads=1, dropout=self.p)

    def forward(self, nodes, edges):
        edges = add_self_loops(nodes.size()[0], edges)
        nodes = F.dropout(nodes, p=self.p, training=self.training)
        x = F.elu(self.conv1(nodes, edges))

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edges)
        return F.log_softmax(x, dim=1)


data = Cora()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= GatNet(1433, 7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
data.get_data_loader([140, 500, 1000], mode='numerical', shuffle=False)
# data.get_data_loader([2000, 300, 400], mode='numerical', shuffle=True)
nodes = data.data_loader.all_node_feature.to(device)
edges = data.data_loader.edge_index.to(device)

all_label = data.data_loader.all_node_label
train_index = data.data_loader.train_index
test_index = data.data_loader.test_index
valid_index = data.data_loader.valid_index
# print(all_label)

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss( model(nodes, edges)[train_index], all_label[train_index] ) + l2_loss(model, 0.001)
    loss.backward()
    optimizer.step()

def l2_loss(model, l2_coef):
    loss_val = 0
    lambda_val = torch.tensor(1.)
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss_val += lambda_val * l2_coef
    return loss_val

def test():
    model.eval()
    logits, accs = model(nodes, edges), []
    for mask in [train_index, valid_index, test_index]:
        # print(logits[mask].size())
        pred = logits[mask].max(1)[1]
        # print('pred = ', pred.size(), pred.eq(all_label[mask]).sum(),  mask.size()[0] )
        acc = pred.eq(all_label[mask]).sum().float() / mask.size()[0]
        accs.append(acc)
    return accs

if __name__ == '__main__':

    # print(nodes.dtype)
    for epoch in range(1, 200):
        train()
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test()))
        # break
