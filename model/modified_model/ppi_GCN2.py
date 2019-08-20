#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/29 10:55

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch.nn.functional as F

import torch
import torch.nn as nn
from model_component.conv.gat_conv_input_list import GatConv
from model_component.conv.gcn_conv_input_list import GCNConv
from torch.nn.parameter import Parameter
from GNN_Implement.model_component.conv.two_order_aggr_conv import TwoOrderAggrConv

class GCN2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.6,
                 bias=True):
        super(GCN2, self).__init__()
        self.mid_layer_channel = 1024
        self.mid_layer_channel2 = self.mid_layer_channel // 2
        self.dropout = dropout
        self.out_channels = out_channels
        self.gcn_conv11 = GCNConv(in_channels,
                            self.mid_layer_channel,
                            dropout=0.6,
                            bias=bias)

        self.gcn_conv12 = GCNConv(in_channels,
                            self.mid_layer_channel,
                            dropout=0.6,
                            bias=bias)

        self.gcn_conv21 = GCNConv(self.mid_layer_channel,
                            self.mid_layer_channel2,
                            dropout=0.6,
                            bias=bias)

        self.gcn_conv22 = GCNConv(self.mid_layer_channel,
                            self.mid_layer_channel2,
                            dropout=0.6,
                            bias=bias)

        self.gcn_conv31 = GCNConv(self.mid_layer_channel2,
                            out_channels,
                            dropout=0.6,
                            bias=bias)

        self.gcn_conv32 = GCNConv(self.mid_layer_channel2,
                            out_channels,
                            dropout=0.6,
                            bias=bias)

        self.aggr_conv1 = TwoOrderAggrConv(self.mid_layer_channel)

        self.aggr_conv2 = TwoOrderAggrConv(self.mid_layer_channel2)

        self.aggr_conv3 = TwoOrderAggrConv(out_channels)

    def forward(self, node_feature, adj_lists):
        one_adj_list, two_adj_list = adj_lists[0], adj_lists[1]
        node_num = node_feature.size()[0]
        node_feature = F.dropout(node_feature,
                                 p=self.dropout,
                                 training=self.training)
        node_state1 = F.elu(self.gcn_conv11(node_feature, one_adj_list))
        node_state2 = F.elu(self.gcn_conv12(node_feature, two_adj_list))
        # 对一阶进行合并
        mid_layer_node_state = self.aggr_conv1(node_state1, node_state2)
        mid_layer_node_state = F.dropout(mid_layer_node_state,
                                 p=self.dropout,
                                 training=self.training)


        node_state1 = F.elu(self.gcn_conv21(mid_layer_node_state, one_adj_list))
        node_state2 = F.elu(self.gcn_conv22(mid_layer_node_state, two_adj_list))
        mid_layer_node_state = self.aggr_conv2(node_state1, node_state2)
        mid_layer_node_state = F.dropout(mid_layer_node_state,
                                 p=self.dropout,
                                 training=self.training)

        node_state1 = F.elu(self.gcn_conv31(mid_layer_node_state, one_adj_list))
        node_state2 = F.elu(self.gcn_conv32(mid_layer_node_state, two_adj_list))
        out = self.aggr_conv3(node_state1, node_state2)
        # print('out = ',out)
        # out = node_state12
        # return F.log_softmax(out, dim=1)
        return out