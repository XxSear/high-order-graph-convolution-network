#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/16 12:35

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
class GatConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.6,
                 bias=True):
        super(GatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_weight1 =  Parameter(
            torch.Tensor(heads * out_channels, 1))

        self.conv_weight2 =  Parameter(
            torch.Tensor(heads * out_channels, 1))


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        # nn.init.xavier_normal_(self.att)
        # init.xavier_normal_(self.bias)
        nn.init.zeros_(self.bias)
        nn.init.xavier_normal_(self.conv_weight1)
        nn.init.xavier_normal_(self.conv_weight2)

    def forward(self, nodes, adj_bias_mat):
        # print(nodes.size(), nodes.dtype, self.weight.dtype)
        # F.dropout(nodes, p=self.dropout, training=True)

        node_hidden_state = torch.mm(nodes, self.weight).view(-1, self.heads * self.out_channels)
        f1 = torch.mm(node_hidden_state, self.conv_weight1)
        f2 = torch.mm(node_hidden_state, self.conv_weight2)
        # print(f1)
        # print(f2)
        logits = f1 + torch.transpose(f2, 0, 1)
        # print(logits.size())
        # print('logits = ',logits)
        # print('adj_bias_mat = ',adj_bias_mat)

        if self.training and self.dropout > 0:
            logits = F.dropout(logits, p=0.2, training=True)
            # adj_bias_mat = F.dropout(logits, p=0.001, training=True)
        coefs = F.softmax(F.leaky_relu(logits, negative_slope=self.negative_slope) + adj_bias_mat, dim=-1)

        # if self.training and self.dropout > 0:
        #     coefs = F.dropout(coefs, p=self.dropout, training=True)
        F.dropout(coefs, p=0.5, training=self.training)
        # print(coefs)
        vals = torch.mm(coefs, node_hidden_state) + self.bias
        # vals = torch.mm(coefs, node_hidden_state)
        # print('vals : ',vals.size(), vals)
        # vals = F.relu(vals)


        # print(vals.size())
        # print(vals)
        # print('self.weight:', self.weight, torch.sum(self.weight, dim=-1))
        return vals



if __name__ == '__main__':
    from data_loader import ACM_3025

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    data = ACM_3025(device=device)

    model = GatConv(
        in_channels=data.node_feature_size,
        out_channels=data.label_size,
        heads=3
        )

    model = model.to(device)
    print(data.adj_mats[1])
    res = model(data.node_feature, data.adj_mats[0])
    print(res)
















