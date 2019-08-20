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
from model_component.conv.bit_gat_list_conv import BitGatConv
from model_test.sample_test import SampleTest

class BitGAT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope=0.28,
                 dropout=0.6,
                 bias=True):
        super(BitGAT, self).__init__()
        self.mid_layer_channel = 8
        self.head = 8
        self.dropout = dropout

        self.gat_conv1 = BitGatConv(in_channels,
                            self.mid_layer_channel,
                            heads=self.head,
                            dropout=dropout,
                            negative_slope=negative_slope,
                            bias=bias)

        self.gat_conv2 = BitGatConv(self.mid_layer_channel * self.head,
                            out_channels,
                            heads=1,
                            dropout=dropout,
                            negative_slope=negative_slope,
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


if __name__ == '__main__':
    model = SampleTest(BitGAT)
    # model.dataset.train_test_split([0.8, 0.1, 0.1], shuffle=True)
    res = model.start()
    print(res)