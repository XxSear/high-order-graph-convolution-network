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
from GNN_Implement.model_component.conv.gat_conv_input_list import GatConv
from torch.nn.parameter import Parameter
from model_component.conv.two_order_aggr_conv import TwoOrderAggrConv

class GAT2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope=0.25,
                 dropout=0.6,
                 bias=True):
        super(GAT2, self).__init__()
        self.mid_layer_channel = 8
        self.head = 8
        self.dropout = dropout
        self.out_channels = out_channels
        self.gat_conv11 = GatConv(in_channels,
                            self.mid_layer_channel,
                            heads=self.head,
                            dropout=0.6,
                            negative_slope=negative_slope,
                            bias=bias)

        self.gat_conv12 = GatConv(self.mid_layer_channel * self.head,
                            out_channels,
                            heads=1,
                            dropout=0.6,
                            negative_slope=negative_slope,
                            bias=bias)

        self.gat_conv21 = GatConv(in_channels,
                            self.mid_layer_channel,
                            heads=self.head,
                            dropout=0.6,
                            negative_slope=negative_slope,
                            bias=bias)

        self.gat_conv22 = GatConv(self.mid_layer_channel * self.head,
                            out_channels,
                            heads=1,
                            dropout=0.6,
                            negative_slope=negative_slope,
                            bias=bias)

        self.aggr_conv1 = TwoOrderAggrConv(self.mid_layer_channel * self.head)

        self.aggr_conv2 = TwoOrderAggrConv(out_channels)


    def forward(self, node_feature, one_adj_list, two_adj_list):
        node_num = node_feature.size()[0]
        node_feature = F.dropout(node_feature,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))
        node_state11 = self.gat_conv11(node_feature, one_adj_list)
        node_state11 = F.elu(node_state11)
        node_state11 = F.dropout(node_state11,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_state)
        node_state21 = self.gat_conv21(node_feature, two_adj_list)
        node_state21 = F.elu(node_state21)
        node_state21 = F.dropout(node_state21,
                                 p=self.dropout,
                                 training=self.training)
        # out = node_state11

        # 对一阶进行合并
        mid_layer_node_state = self.aggr_conv1(node_state11, node_state21)
        # out = mid_layer_node_state
        # print('mid_layer_node_state = ',mid_layer_node_state)
        mid_layer_node_state = F.dropout(mid_layer_node_state, p=self.dropout, training=self.training)
        node_state12 = self.gat_conv12(mid_layer_node_state, one_adj_list)
        node_state22 = self.gat_conv22(mid_layer_node_state, two_adj_list)
        # node_state12 = F.elu(node_state12)
        # node_state22 = F.elu(node_state22)
        node_state12 = F.dropout(node_state12,
                                 p=self.dropout,
                                 training=self.training)
        node_state22 = F.dropout(node_state22,
                                 p=self.dropout,
                                 training=self.training)

        out = self.aggr_conv2(node_state12, node_state22)
        # print('out = ',out)
        # out = node_state12
        return F.log_softmax(out, dim=1)

