#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/11 16:32

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
import torch.nn.functional as F

import torch
import torch.nn as nn
from model_component.conv.ggnn_conv_input_list import GGNNConv


# 无向图  node_type = 1
class GGNN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.6,
                 propagate_step = 2,
                 ):
        super(GGNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.propagate_step = propagate_step

        self.ggnn_conv = GGNNConv(
            in_channels,
            dropout=self.dropout,
            propagate_step=self.propagate_step,
        )

        # 在输出阶段 有两个神经网络 i, j
        self.mlp_i = nn.Sequential(
            nn.Linear(self.in_channels*2, self.in_channels),
            nn.Linear(self.in_channels, self.out_channels),
            nn.Sigmoid()
        )
        self.mlp_j = nn.Sequential(
            nn.Linear(self.in_channels*2, self.in_channels),
            nn.Linear(self.in_channels, self.out_channels),
            nn.Tanh()
        )

    def forward(self, nodes_ft, adj_list):

        adj_state = self.ggnn_conv(nodes_ft, adj_list)

        # print(adj_state)
        concat_state = torch.cat((adj_state, nodes_ft), -1)
        # out = self.mlp_i(concat_state) + self.mlp_j(concat_state)
        # out = torch.tanh(out)


        out = self.mlp_i(concat_state)

        return out
        # return F.log_softmax(out, dim=-1)


if __name__ == '__main__':
    from GNN_Implement.data_loader import Cora

    dataset = Cora()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset.to_device(device)


    ggnn = GGNN(
        in_channels=dataset.node_feature_size,
        out_channels=dataset.label_size
    ).to(device)

    res =  ggnn(dataset.all_node_feature, dataset.add_self_loop_edge_list)