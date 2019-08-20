#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/11 15:44

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from data_loader import Cora
from model_component.utils.agg_softmax import softmax
from model_component.utils.scatter import scatter_
from model_component.utils.add_self_loop import list_add_self_loops
from torch_scatter import scatter_add

# GRU layer
# 无向图  node_type = 1
class GGNNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 dropout=0.6,
                 propagate_step = 4,
                 ):
        super(GGNNConv, self).__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.propagate_step = propagate_step

        self.nodes_adj_state_bias = Parameter(
            torch.Tensor(1, in_channels)
        ) # 所有节点共享一个bias
        nn.init.xavier_normal_(self.nodes_adj_state_bias)

        self.reset_gate = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.Tanh()
        )



    def forward(self, nodes_ft, adj_list):
        prior_h = nodes_ft
        # 传播
        # print(nodes_ft.size(), adj_list.size())
        for i_step in range(self.propagate_step):
            # 单个节点对邻居加权求和
            nodes_adj_state = torch.index_select(prior_h, 0, adj_list[1])
            # print('index_select = ', nodes_adj_state.size())
            nodes_adj_state = scatter_('add', nodes_adj_state, adj_list[0], dim_size=prior_h.size()[0])
            # print('scatter_ = ', nodes_adj_state.size())
            nodes_adj_state = nodes_adj_state + self.nodes_adj_state_bias
            nodes_adj_state = F.softmax(nodes_adj_state, dim=-1)

            # GRU
            tmp_state = torch.cat((nodes_adj_state, prior_h), -1)
            r = self.reset_gate(tmp_state)
            z = self.update_gate(tmp_state)
            h_hat = self.tansform(
                torch.cat((nodes_adj_state, r * prior_h), -1)
            )
            h = (1 - z) * prior_h + z * h_hat

        return h

