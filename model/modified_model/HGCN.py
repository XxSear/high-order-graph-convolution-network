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

from model_component.conv.gcn_conv_input_list import GCNConv
from torch.nn.parameter import Parameter
from model_component.conv.two_order_aggr_conv import TwoOrderAggrConv
from model_component.conv.n_order_aggr_conv import NOrderAggrConv

class SturtGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.6,
                 bias=True,
                 layer=2,
                 mid_layer_channel=128,
                 order=3,
                 aggr_depth=2,
                 aggr_type=''
                 ):
        super(SturtGCN, self).__init__()
        self.mid_layer_channel = mid_layer_channel
        self.dropout = dropout
        self.num_layer = layer
        self.out_channels = out_channels
        self.order = order
        self.bias = bias

        # self.n_order_gcn_conv_list = [ [] for i in range(self.num_layer) ]
        self.n_order_gcn_conv_list = nn.ModuleList()
        self.n_order_agg_conv_list = nn.ModuleList()
        for gcn_depth in range(self.num_layer):
            input_size = self.mid_layer_channel
            out_size = self.mid_layer_channel
            if gcn_depth == 0:
                input_size = in_channels
            if gcn_depth == self.num_layer - 1:
                out_size = out_channels

            tmp_layer_gcn_conv = nn.ModuleList()
            for order_idx in range(self.order):
                tmp_layer = GCNConv(
                    input_size,
                    out_size,
                    dropout=self.dropout,
                    bias=self.bias)
                tmp_layer_gcn_conv.append(tmp_layer)
            self.n_order_gcn_conv_list.append(tmp_layer_gcn_conv)

            if gcn_depth != self.num_layer - 1:
                tmp_agg_conv = NOrderAggrConv(self.mid_layer_channel, n=order, depth=aggr_depth)
            else:
                tmp_agg_conv = NOrderAggrConv(self.out_channels, n=order, depth=aggr_depth)
            self.n_order_agg_conv_list.append(tmp_agg_conv)

        # print(len(self.n_order_gcn_conv_list), len(self.n_order_agg_conv_list))
        # print(self.n_order_agg_conv_list)
        # print(self.n_order_gcn_conv_list)

    def forward(self, node_feature, n_order_adj_list):
        node_num = node_feature.size()[0]
        node_feature = F.dropout(node_feature,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))

        node_state = node_feature
        n_order_node_state = [ i for i in range(self.order)]
        for depth_idx in range(self.num_layer):
            for order_idx in range(self.order):
                tmp_node_state = self.n_order_gcn_conv_list[depth_idx][order_idx](node_state, n_order_adj_list[order_idx])
                tmp_node_state = F.elu(tmp_node_state)
                tmp_node_state = F.dropout(tmp_node_state,
                                     p=self.dropout,
                                     training=self.training)
                n_order_node_state[order_idx] = tmp_node_state

            node_state = self.n_order_agg_conv_list[depth_idx](n_order_node_state)
            # if depth_idx != self.num_layer - 1:
            #     node_state = F.dropout(node_state, p=self.dropout, training=self.training)

        return F.log_softmax(node_state, dim=1)

