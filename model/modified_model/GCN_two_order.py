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
from model_component.conv.two_order_aggr_conv import TwoOrderAggrConv


class GCNTwoOrder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 imporve=False,
                 dropout=0.6,
                 bias=True):
        super(GCNTwoOrder, self).__init__()
        self.mid_layer_channel = 16
        self.dropout = dropout

        self.conv11 = GCNConv(in_channels,
                            self.mid_layer_channel,
                            dropout=self.dropout,
                            bias=bias)

        self.conv12 = GCNConv(self.mid_layer_channel,
                            out_channels,
                            dropout=self.dropout,
                            bias=bias)

        self.conv21 = GCNConv(in_channels,
                            self.mid_layer_channel,
                            dropout=self.dropout,
                            bias=bias)

        self.conv22 = GCNConv(self.mid_layer_channel,
                            out_channels,
                            dropout=self.dropout,
                            bias=bias)

        self.combine_conv = TwoOrderAggrConv(out_channels, dropout=dropout)

    def forward(self, node_feature, adj_list, two_order_adj_list):
        # node_feature = F.dropout(node_feature,
        #                          p=self.dropout,
        #                          training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))

        # one_order
        node_state = self.conv11(node_feature, adj_list)
        node_state = F.elu(node_state)
        node_state = F.dropout(node_state,
                                 p=self.dropout,
                                 training=self.training)
        first_order_coding = self.conv12(node_state, adj_list)

        # two_order
        # print(node_feature.size(), two_order_adj_list.size(), adj_list.size())
        two_order_node_state = self.conv21(node_feature, two_order_adj_list)
        two_order_node_state = F.elu(two_order_node_state)
        two_order_node_state = F.dropout(two_order_node_state,
                                 p=self.dropout,
                                 training=self.training)
        second_order_coding = self.conv22(two_order_node_state, two_order_adj_list)

        out = self.combine_conv(first_order_coding, second_order_coding)

        # return F.log_softmax(first_order_coding, dim=1)
        # return F.log_softmax(second_order_coding, dim=1)
        return F.log_softmax(out, dim=1)