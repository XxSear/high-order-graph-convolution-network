#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/8 15:29

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import torch.nn.functional as F
from model_component.conv.gat_conv import GAT
from torch.nn import Parameter
from data_loader import Cora


class Net(torch.nn.Module):
    def __init__(self, node_feature_size, num_classes, att_out_size, heads=1, dropout=0.6, concat=True):
        super(Net, self).__init__()
        self.gat_conv = GAT(node_feature_size, att_out_size, heads=heads, dropout=dropout)
        if concat is True:
            self.weight = Parameter(torch.empty([att_out_size * heads, num_classes]), requires_grad=True)
        else:
            self.weight = Parameter(torch.empty([att_out_size, num_classes]), requires_grad=True)

        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, node_feature, neighbor_nodes_feature):
        x = self.gat_conv(node_feature, neighbor_nodes_feature)  #[1, att_out_size]
        x = torch.mm(x, self.weight)  #
        # print(x)
        # out = F.softmax(x)
        # print(res)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(1433, 7, 32).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data = Cora()
train_data_loader, valid_data_loader, test_data_loader = data.get_data_loader([4,2,4])
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.functional.nll_loss()

def train():
    for epoch in range(100):
        total_loss = 0
        for i, batch_data in enumerate(train_data_loader, 0):
            center_node_index,center_node_feature, center_node_label, neighbor_nodes_feature = batch_data
            # print(center_node_feature[0].size(), neighbor_nodes_feature[0].size())

            res = model(center_node_feature, neighbor_nodes_feature[0])
            # print(center_node_label, res)
            # loss = criterion(res, center_node_label)
            loss = F.nll_loss(F.log_softmax(res), center_node_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 500 == 0:
                # print('loss = ',total_loss)
                total_loss = 0
            # exit()
        test()
        # for a in model.state_dict():
        # for a in model.named_parameters():
        #     print(a)

def test():
    total = 0
    correct = 0
    model.eval()
    for i, batch_data in enumerate(test_data_loader, 0):
        center_node_index,center_node_feature, center_node_label, neighbor_nodes_feature = batch_data
        predict = model(center_node_feature, neighbor_nodes_feature[0])
        pred_y = torch.max(predict, 1)[1].cpu().data.numpy()
        real_y = center_node_label.cpu().data.numpy()
        # print(pred_y, predict)
        # print(real_y)
        # exit()
        total += 1
        if pred_y[0] == real_y[0]:
            correct+= 1
        # print(pred_y, real_y)
        # break
    print('acc = ', 100 * correct / total)

train()
# test()
# print(data.data_loader.node_label)
