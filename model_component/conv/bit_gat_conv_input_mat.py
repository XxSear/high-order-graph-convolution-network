#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/11 17:06

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class BitGatConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.6,
                 bias=True):
        super(BitGatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_weight1 =  Parameter(
            torch.Tensor(heads * out_channels, heads * out_channels))

        self.conv_weight2 =  Parameter(
            torch.Tensor(heads * out_channels, heads * out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.att)
        # init.xavier_normal_(self.bias)
        nn.init.zeros_(self.bias)
        nn.init.xavier_normal_(self.conv_weight1)
        nn.init.xavier_normal_(self.conv_weight2)

    def forward(self, nodes_ft, adj_bias_mat):
        n = nodes_ft.size()[0]
        node_hidden_state = torch.mm(nodes_ft, self.weight).view(-1, self.heads * self.out_channels)
        f1 = torch.mm(node_hidden_state, self.conv_weight1).view(n, 1, self.heads * self.out_channels)
        f2 = torch.mm(node_hidden_state, self.conv_weight2).view(n, 1, self.heads * self.out_channels)

        logits = f1 + torch.transpose(f2, 0, 1)

        coefs = F.softmax(F.leaky_relu(logits, negative_slope=self.negative_slope) + adj_bias_mat.view(n, n, 1), dim=1)
        if self.training and self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=True)

        vals = coefs * node_hidden_state.view(n, 1, self.heads * self.out_channels)
        vals = torch.sum(vals, dim=1)

        return vals

