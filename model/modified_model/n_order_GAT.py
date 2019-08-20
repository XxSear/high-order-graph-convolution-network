#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/30 16:17

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch.nn.functional as F

import torch
import torch.nn as nn


from model_component.conv.gat_conv_input_list import GatConv

from torch.nn.parameter import Parameter
from model_component.conv.two_order_aggr_conv import TwoOrderAggrConv
from model_component.conv.n_order_aggr_conv import NOrderAggrConv

class SturtGAT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.6,
                 bias=True,
                 layer=2,
                 mid_layer_channel=128,
                 heads=2,
                 order=3,
                 aggr_depth=2,
                 ):
        super(SturtGAT, self).__init__()
        self.mid_layer_channel = mid_layer_channel
        self.dropout = dropout
        self.num_layer = layer
        self.out_channels = out_channels
        self.order = order
        self.bias = bias
        self.heads = heads

        # self.n_order_GAT_conv_list = [ [] for i in range(self.num_layer) ]
        self.n_order_gat_conv_list = nn.ModuleList()
        self.n_order_agg_conv_list = nn.ModuleList()
        for gat_depth in range(self.num_layer):
            input_size = self.mid_layer_channel * self.heads
            out_size = self.mid_layer_channel
            if gat_depth == 0:
                input_size = in_channels
            if gat_depth == self.num_layer - 1:
                out_size = out_channels

            if gat_depth == self.num_layer - 1:
                concat_flag = False
            else:
                concat_flag = True

            tmp_layer_gat_conv = nn.ModuleList()
            for order_idx in range(self.order):
                tmp_layer = GatConv(
                    input_size,
                    out_size,
                    dropout=self.dropout,
                    bias=self.bias,
                    heads=self.heads,
                    concat=concat_flag
                )
                tmp_layer_gat_conv.append(tmp_layer)
            self.n_order_gat_conv_list.append(tmp_layer_gat_conv)

            if gat_depth != self.num_layer - 1:
                tmp_agg_conv = NOrderAggrConv(self.mid_layer_channel * self.heads, n=order, depth=aggr_depth)
            else:
                tmp_agg_conv = NOrderAggrConv(self.out_channels, n=order, depth=aggr_depth)
            self.n_order_agg_conv_list.append(tmp_agg_conv)

        # print(len(self.n_order_gat_conv_list), len(self.n_order_agg_conv_list))
        # print(self.n_order_agg_conv_list)
        # print(self.n_order_gat_conv_list)

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
                tmp_node_state = self.n_order_gat_conv_list[depth_idx][order_idx](node_state, n_order_adj_list[order_idx])
                tmp_node_state = F.elu(tmp_node_state)
                # tmp_node_state = F.dropout(tmp_node_state,
                #                  p=self.dropout,
                #                  training=self.training)
                n_order_node_state[order_idx] = tmp_node_state

            # print('depth_idx = ', depth_idx)
            node_state = self.n_order_agg_conv_list[depth_idx](n_order_node_state)
            if depth_idx != self.num_layer - 1:
                node_state = F.dropout(node_state, p=self.dropout, training=self.training)

        return F.log_softmax(node_state, dim=1)

