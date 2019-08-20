#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/17 9:18

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch.nn.functional as F

import torch
import torch.nn as nn
from model_component.conv.gcn_conv_input_list import GCNConv


class GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 imporve=False,
                 dropout=0.5,
                 bias=True):
        super(GCN, self).__init__()
        self.mid_layer_channel = 64
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels,
                            self.mid_layer_channel,
                            bias=bias)

        self.conv2 = GCNConv(self.mid_layer_channel,
                            out_channels,
                            bias=bias)

    def forward(self, node_feature, adj_list):
        # node_feature = F.dropout(node_feature,
        #                          p=self.dropout,
        #                          training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))
        node_state = self.conv1(node_feature, adj_list)
        node_state = F.elu(node_state)
        # node_state = F.dropout(node_state,
        #                          p=self.dropout,
        #                          training=self.training)
        out = self.conv2(node_state, adj_list)
        # print(out.size(), F.log_softmax(out, dim=1).size())
        return F.log_softmax(out, dim=1)
        # return out