#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/14 18:05

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

'''
    使用邻接矩阵作为输入 当结点数较大时 显存不够
'''

import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from data_loader import Cora
from model_component.utils.agg_softmax import softmax
from model_component.utils.scatter import scatter_

class GatConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=4,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.att_layer =  Parameter(
            torch.Tensor(1, heads, out_channels * 2))



        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        # nn.init.xavier_normal_(self.att)
        # init.xavier_normal_(self.bias)
        nn.init.zeros_(self.bias)
        nn.init.xavier_normal_(self.att_layer)


    def forward(self, nodes_ft, adj_list):
        # adj_list 是加过 self-loop 的列表
        edge_num = adj_list.size()[1]
        # print(edge_num)
        node_hidden_state = torch.mm(nodes_ft, self.weight).view(-1, self.heads, self.out_channels)

        egde_i_node_state = torch.index_select(node_hidden_state, 0, adj_list[0]).view(edge_num, self.heads, self.out_channels)
        egde_j_node_state = torch.index_select(node_hidden_state, 0, adj_list[1]).view(edge_num, self.heads, self.out_channels)
        edge_node_state = torch.cat([egde_i_node_state, egde_j_node_state], dim=-1)

        # print(edge_node_state.size(), self.att_layer.size())
        alpha = (edge_node_state * self.att_layer).sum(dim=-1)

        # logits_list = torch.cat((logits_list_i, logits_list_j), -1).sum(-1)
        logits_list = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        # print(logits_list.size())

        # 对应位置做softmax
        alpha = softmax(logits_list, adj_list[0], nodes_ft.size()[0])
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        # print('alpha.size() = ',alpha.size(), adj_list.size())

        vals = egde_j_node_state * alpha.view(edge_num, self.heads, 1)
        out_node_state = scatter_('add', vals, adj_list[0], dim_size=nodes_ft.size()[0])

        if self.concat is True:
            out_node_state = out_node_state.view(-1, self.heads * self.out_channels)
        else:
            out_node_state = out_node_state.mean(dim=1)

        if self.bias is not None:
            # print('out_node_state size = ', out_node_state.size(), 'bias size = ', self.bias.size())
            out_node_state = out_node_state + self.bias

        # print('out_node_state size = ', out_node_state.size())
        return out_node_state

if __name__ == '__main__':
    dataset = Cora()
    net = GatConv(
        dataset.node_feature_size,
        dataset.label_size,
    )
    net(dataset.all_node_feature, dataset.edge_index)