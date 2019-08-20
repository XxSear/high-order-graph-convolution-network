#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/25 16:10

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch.nn.functional as F

import torch
import torch.nn as nn
from GNN_Implement.model_component.conv.gcn_conv_input_list import GCNConv
from GNN_Implement.model_component.utils.add_self_loop import list_add_self_loops

class GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channel=1024,
                 imporve=False,
                 dropout=0.5,
                 bias=True):
        super(GCN, self).__init__()
        self.input_channel = in_channels
        self.out_channels = out_channels
        self.mid_layer_channel = mid_channel
        self.mid_layer_channel2 = mid_channel // 2
        self.dropout = dropout

        self.gcn_conv1 = GCNConv(in_channels,
                            self.mid_layer_channel,
                            bias=bias)

        self.lin1 = torch.nn.Linear(self.input_channel , self.mid_layer_channel)

        self.gcn_conv2 = GCNConv(self.mid_layer_channel,
                            self.mid_layer_channel2,
                            bias=bias)
        self.lin2 = torch.nn.Linear(self.mid_layer_channel, self.mid_layer_channel2)

        self.gcn_conv3 = GCNConv(self.mid_layer_channel2,
                            self.out_channels,
                            bias=bias)
        self.lin3 = torch.nn.Linear(self.mid_layer_channel2, self.out_channels)


    def forward(self, node_feature, adj_list):
        # node_feature = F.dropout(node_feature,
        #                          p=self.dropout,
        #                          training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))
        adj_list = list_add_self_loops(node_feature.size()[0], adj_list)
        node_state = F.elu(self.gcn_conv1(node_feature, adj_list)) + self.lin1(node_feature)


        # node_state = F.dropout(node_state,
        #                          p=self.dropout,
        #                          training=self.training)
        node_state = F.elu(self.gcn_conv2(node_state, adj_list)) + self.lin2(node_state)

        # node_state = F.dropout(node_state,
        #                        p=self.dropout,
        #                        training=self.training)
        out = self.gcn_conv3(node_state, adj_list) + self.lin3(node_state)

        # print(out.size(), F.log_softmax(out, dim=1).size())
        # return F.log_softmax(out, dim=1)
        return out