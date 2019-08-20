#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/29 14:16

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

class GAT2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0,
                 heads=4,
                 bias=True):
        super(GAT2, self).__init__()
        self.mid_layer_channel = 128
        self.mid_layer_channel2 = self.mid_layer_channel // 2
        self.dropout = dropout
        self.heads = heads
        self.out_channels = out_channels
        self.gcn_conv11 = GatConv(in_channels,
                            self.mid_layer_channel,
                            heads=self.heads,
                            dropout=self.dropout,
                            bias=bias)

        self.gcn_conv12 = GatConv(in_channels,
                                self.mid_layer_channel,
                                heads=self.heads,
                                dropout=self.dropout,
                                bias=bias)

        self.gcn_conv21 = GatConv(self.mid_layer_channel * self.heads,
                            self.mid_layer_channel2,
                                  heads=self.heads,
                            dropout=self.dropout,
                            bias=bias)

        self.gcn_conv22 = GatConv(self.mid_layer_channel * self.heads,
                            self.mid_layer_channel2,
                                  heads=self.heads,
                            dropout=self.dropout,
                            bias=bias)

        self.gcn_conv31 = GatConv(self.mid_layer_channel2 * self.heads,
                                  out_channels,
                                  heads=self.heads,
                                  dropout=self.dropout,
                                  concat=False,
                                  bias=bias)

        self.gcn_conv32 = GatConv(self.mid_layer_channel2 * self.heads,
                                  out_channels,
                                  heads=self.heads,
                                  dropout=self.dropout,
                                  concat=False,
                                  bias=bias)

        self.aggr_conv1 = TwoOrderAggrConv(self.mid_layer_channel * self.heads)

        self.aggr_conv2 = TwoOrderAggrConv(self.mid_layer_channel2 * self.heads)

        self.aggr_conv3 = TwoOrderAggrConv(out_channels)

        self.lin1 = torch.nn.Linear(in_channels, self.mid_layer_channel * self.heads)
        self.lin2 = torch.nn.Linear(self.mid_layer_channel * self.heads, self.mid_layer_channel2 * self.heads)
        self.lin3 = torch.nn.Linear(self.mid_layer_channel2 * self.heads, out_channels)



    def forward(self, node_feature, adj_lists):
        one_adj_list, two_adj_list = adj_lists[0], adj_lists[1]
        node_num = node_feature.size()[0]

        node_state1 = F.elu(self.gcn_conv11(node_feature, one_adj_list))
        node_state2 = F.elu(self.gcn_conv12(node_feature, two_adj_list))
        # 对一阶进行合并
        # print(node_state1.size(), node_state1.size())
        mid_layer_node_state = self.aggr_conv1(node_state1, node_state2) + self.lin1(node_feature)
        # mid_layer_node_state = F.dropout(mid_layer_node_state,
        #                          p=self.dropout,
        #                          training=self.training)


        node_state1 = F.elu(self.gcn_conv21(mid_layer_node_state, one_adj_list))
        node_state2 = F.elu(self.gcn_conv22(mid_layer_node_state, two_adj_list))
        mid_layer_node_state = self.aggr_conv2(node_state1, node_state2) + self.lin2(mid_layer_node_state)
        # mid_layer_node_state = F.dropout(mid_layer_node_state,
        #                          p=self.dropout,
        #                          training=self.training)

        node_state1 = F.elu(self.gcn_conv31(mid_layer_node_state, one_adj_list))
        node_state2 = F.elu(self.gcn_conv32(mid_layer_node_state, two_adj_list))
        out = self.aggr_conv3(node_state1, node_state2) + self.lin3(mid_layer_node_state)
        # print('out = ',out)
        # out = node_state12
        # return F.log_softmax(out, dim=1)
        return out