#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/26 13:43

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

from GNN_Implement.model_component.conv.gat_conv_input_list import GatConv
from GNN_Implement.model_component.utils.add_self_loop import list_add_self_loops
import torch
import torch.nn.functional as F
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope=0.28,
                 dropout=0.6,
                 bias=True):
        super(GAT, self).__init__()
        self.mid_layer_channel = 256
        self.head = 4
        self.dropout = dropout

        self.conv1 = GatConv(in_channels, 256, heads=4)
        self.lin1 = torch.nn.Linear(in_channels, 4 * 256)
        self.conv2 = GatConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GatConv(
            4 * 256, out_channels, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, out_channels)

    def forward(self, x, edge_index):
        # print(x.size()[0])
        edge_index = list_add_self_loops(x.size()[0], edge_index)
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x