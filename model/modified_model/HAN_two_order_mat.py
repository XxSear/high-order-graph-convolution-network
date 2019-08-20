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
from model_component.conv.gat_list_as_input import GatConv
from torch.nn.parameter import Parameter
from model_component.conv.aggregation_attn_conv import AggregatioAttnConv

class TwoOrderHat(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope=0.25,
                 dropout=0.6,
                 bias=True):
        super(TwoOrderHat, self).__init__()
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

        self.aggr_conv = AggregatioAttnConv(out_channels, out_channels)

        self.aggr_trans = Parameter(
            torch.Tensor(out_channels, out_channels))
        self.aggr_att = Parameter(
            torch.Tensor(1, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.combine_weight = Parameter(
            torch.Tensor(1, out_channels, 2))
        self.combine_weight2 = Parameter(
            torch.Tensor(1, out_channels, 2))
        # self.combine_bias = Parameter(
        #     torch.Tensor(out_channels))


        self.one_layer_gat_trans1 = Parameter(
            torch.Tensor(self.mid_layer_channel * self.head, out_channels))

        self.one_layer_gat_trans2 = Parameter(
            torch.Tensor(self.mid_layer_channel * self.head, out_channels))

        nn.init.xavier_normal_(self.one_layer_gat_trans1)
        nn.init.xavier_normal_(self.one_layer_gat_trans2)

        nn.init.xavier_normal_(self.combine_weight)
        nn.init.xavier_normal_(self.combine_weight2)
        nn.init.xavier_normal_(self.aggr_trans)
        nn.init.xavier_normal_(self.aggr_att)
        # nn.init.xavier_normal_(self.aggr_trans)
        nn.init.zeros_(self.bias)

    def forward(self, node_feature, one_order_bias, two_order_bias):
        node_num = node_feature.size()[0]
        node_feature = F.dropout(node_feature,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_feature.size(), torch.sum(node_feature, dim=-1))
        node_state11 = self.gat_conv11(node_feature, one_order_bias)
        node_state11 = F.elu(node_state11)
        node_state11 = F.dropout(node_state11,
                                 p=self.dropout,
                                 training=self.training)
        # print(node_state)
        node_state21 = self.gat_conv21(node_feature, two_order_bias)
        node_state21 = F.elu(node_state21)
        node_state21 = F.dropout(node_state21,
                                 p=self.dropout,
                                 training=self.training)

        # 只用单层GAT
        # node_state12 = torch.mm(node_state11, self.one_layer_gat_trans1)
        # node_state22 = torch.mm(node_state21, self.one_layer_gat_trans1)

        # print('node_state = ',node_state)
        node_state12 = self.gat_conv12(node_state11, one_order_bias)
        node_state22 = self.gat_conv22(node_state21, two_order_bias)

        node_state12 = F.elu(node_state12)
        node_state22 = F.elu(node_state22)
        # node_state12 = torch.tanh(node_state12)
        # node_state22 = torch.tanh(node_state22)

        # F.dropout(node_state22,
        #           p=0.3,
        #           training=True)
        # F.dropout(node_state12,
        #           p=0.3,
        #           training=True)
        # combine_state = torch.cat((node_state12,node_state22), -1)
        # combine_state = torch.mm(combine_state, self.combine_layer)


        combine_state = torch.cat((node_state12.view(node_num, self.out_channels, 1),
                                   node_state22.view(node_num, self.out_channels, 1)), -1)
        combine_state = combine_state * self.combine_weight
        # F.dropout(combine_state, p=0.6, training=True)
        F.dropout(combine_state, p=0.6, training=self.training)
        combine_state = F.elu(combine_state)
        combine_state = combine_state * self.combine_weight2
        combine_state = torch.sum(combine_state, -1)
        # combine_state = torch.mm(combine_state, self.combine_weight).view(node_num, self.out_channels)
        # combine_state = (node_state22 + node_state12) / 2
        # print('combine_weight = ',self.combine_weight)

        # combine_state = torch.cat([node_state12.view(node_num, 1, self.out_channels),
        #                            node_state22.view(node_num, 1, self.out_channels)], 0)
        # print(combine_state.size())
        # combine_state, _ = self.aggr_conv(combine_state)
        # combine_state = torch.mm(combine_state.view(node_num * 2, -1), self.aggr_trans) + self.bias
        # combine_state = combine_state.view(node_num, 2, -1)
        # w = (self.aggr_att * torch.tanh(combine_state)).sum(dim=-1)
        # w = F.softmax(w, dim=1)
        # print(w)
        # out = combine_state * (w.view(node_num, 2, 1))
        # out = torch.transpose(out, 1, 2)
        # out = out.sum(-1)
        # print(_)
        # out = (node_state12 + node_state22) / 2
        return F.log_softmax(combine_state, dim=1)
        # return F.log_softmax(node_state12, dim=1)
        # return F.log_softmax(node_state22, dim=1)

