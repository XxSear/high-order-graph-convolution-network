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
from model_component.conv.gat_conv_input_list import GatConv


class GAT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope=0.28,
                 dropout=0.6,
                 heads=8,
                 bias=True):
        super(GAT, self).__init__()
        self.mid_layer_channel = 8
        self.heads = heads
        self.dropout = dropout

        self.gat_conv1 = GatConv(in_channels,
                            self.mid_layer_channel,
                            heads=self.heads,
                            dropout=dropout,
                            negative_slope=negative_slope,
                            concat=True,
                            bias=bias)

        self.gat_conv2 = GatConv(self.mid_layer_channel * self.heads,
                            out_channels,
                            heads=heads,
                            dropout=dropout,
                            negative_slope=negative_slope,
                                concat=False,
                            bias=bias)

    def forward(self, node_feature, adj_list):
        node_feature = F.dropout(node_feature,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))
        node_state = self.gat_conv1(node_feature, adj_list)

        node_state = F.elu(node_state)
        # print(node_state)

        node_state = F.dropout(node_state,
                                 p=self.dropout,
                                 training=self.training)
        # print('node_state = ',node_state)
        out = self.gat_conv2(node_state, adj_list)
        # print('out = ',out)
        return F.log_softmax(out, dim=1)

