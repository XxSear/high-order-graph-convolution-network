#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/25 22:02

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
import torch.nn.functional as F

import torch
import torch.nn as nn
from model_component.conv.gat_mat_as_input import GatConv


class GAT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope=0.28,
                 dropout=0.6,
                 bias=True):
        super(GAT, self).__init__()
        self.mid_layer_channel = 8
        self.head = 8
        self.dropout = dropout

        self.gat_conv1 = GatConv(in_channels,
                            self.mid_layer_channel,
                            heads=self.head,
                            dropout=dropout,
                            negative_slope=negative_slope,
                            bias=bias)

        self.gat_conv2 = GatConv(self.mid_layer_channel * self.head,
                            out_channels,
                            heads=1,
                            dropout=dropout,
                            negative_slope=negative_slope,
                            bias=bias)

    def forward(self, node_feature, bias_adj_mat):
        node_feature = F.dropout(node_feature,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))
        node_state = self.gat_conv1(node_feature, bias_adj_mat)

        node_state = F.elu(node_state)
        # print(node_state)

        node_state = F.dropout(node_state,
                                 p=self.dropout,
                                 training=self.training)
        # print('node_state = ',node_state)
        out = self.gat_conv2(node_state, bias_adj_mat)
        # print('out = ',out)
        return F.log_softmax(out, dim=1)

