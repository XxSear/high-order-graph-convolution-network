#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/5 22:21

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
import torch
import torch.nn as nn
from model_component.conv.gat_mat_as_input import GatConv
import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from GNN_Implement.model_component.conv.two_order_gat_conv import TwoOrderGatConv


class TwoOrderGat(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.6,
                 bias=True):
        super(TwoOrderGat, self).__init__()
        self.mid_state_size = 8
        self.first_heads = 8
        self.dropout = dropout
        self.gat_conv1 = TwoOrderGatConv(in_channels,
                                         self.mid_state_size,
                                         heads=self.first_heads,
                                         negative_slope=negative_slope,
                                         dropout=dropout)
        self.gat_conv2 = TwoOrderGatConv(self.mid_state_size * self.first_heads,
                                         out_channels,
                                         heads=1,
                                         negative_slope=negative_slope,
                                         dropout=dropout)
        self.out_weight = Parameter(torch.Tensor(self.mid_state_size * self.first_heads, out_channels))
        nn.init.xavier_normal_(self.out_weight)

    def forward(self, nodes_ft, one_order_adj_bias_mat, two_order_adj_bias_mat, l2_norm):

        nodes_ft = F.dropout(nodes_ft,
                             p=self.dropout,
                             training=self.training)
        node_state = self.gat_conv1(nodes_ft, one_order_adj_bias_mat, two_order_adj_bias_mat, l2_norm)
        node_state = F.elu(node_state)

        # out = torch.mm(node_state, self.out_weight)
        node_state = F.dropout(node_state,
                             p=self.dropout,
                             training=self.training)
        out = self.gat_conv2(node_state, one_order_adj_bias_mat, two_order_adj_bias_mat, l2_norm)

        return F.log_softmax(out, dim=1)