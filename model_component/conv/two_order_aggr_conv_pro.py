#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/17 10:58

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F


class TwoOrderAggrConv(torch.nn.Module):
    def __init__(self, in_channels, dropout=0.6, trans=True, bit_layer=1):
        super(TwoOrderAggrConv, self).__init__()
        self.dropout = 0.5
        self.in_channels = in_channels
        self.trans = trans

        self.combine_weight_layer1 = Parameter(
            torch.Tensor(1, in_channels, 2))
        init.xavier_normal_(self.combine_weight_layer1)

        self.combine_weight_layer2 = Parameter(
            torch.Tensor(1, in_channels, 2))
        init.xavier_normal_(self.combine_weight_layer2)


    def forward(self, first_order_vec, second_order_vec):
        # first_order_vec = F.elu(first_order_vec)
        # second_order_vec = F.elu(second_order_vec)
        # if self.trans:
        # first_order_vec = F.dropout(first_order_vec, p=0.6, training=self.training)
        # second_order_vec = F.dropout(second_order_vec, p=0.6, training=self.training)
        node_num = first_order_vec.size()[0]


        combine_state = torch.cat((first_order_vec.view(node_num, self.in_channels, 1),
                                   second_order_vec.view(node_num, self.in_channels, 1)), -1)
        combine_state = combine_state * self.combine_weight_layer1
        # combine_state = F.dropout(combine_state, p=0.5, training=self.training)
        # combine_state = F.elu(combine_state)
        # combine_state = combine_state * self.combine_weight_layer2
        combine_state = torch.sum(combine_state, -1)
        # combine_state = torch.max(combine_state, dim=-1)[0]

        return combine_state