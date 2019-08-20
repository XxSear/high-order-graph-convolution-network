#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/8/1 14:47

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


from GNN_Implement.model_component.conv.gat_conv_input_list import GatConv
from GNN_Implement.model_component.utils.add_self_loop import list_add_self_loops
from GNN_Implement.model_component.conv.igcn_conv_input_mat import IGCNConv
import torch
import torch.nn.functional as F
import torch.nn as nn

class IGCN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 k=2,
                 alpha=10,
                 model_type='rnm',
                 imporve=False,
                 dropout=0.5,
                 bias=True):
        super(IGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.alpha=alpha
        self.model_type = model_type
        self.dropout = dropout
        self.bias = bias
        self.mid_channels = 256

        self.igcn_conv1 = IGCNConv(self.in_channels, self.mid_channels,
                                   bias=self.bias, k=self.k, type=self.model_type, alpha=self.alpha)
        self.igcn_conv2 = IGCNConv(self.mid_channels,  self.mid_channels,
                                   bias=self.bias, k=self.k, type=self.model_type, alpha=self.alpha)
        self.igcn_conv3 = IGCNConv(self.mid_channels, self.out_channels,
                                   bias=self.bias, k=self.k, type=self.model_type, alpha=self.alpha)


        self.lin1 = torch.nn.Linear(self.in_channels, self.mid_channels)

        self.lin2 = torch.nn.Linear(self.mid_channels, self.mid_channels)

        self.lin3 = torch.nn.Linear(self.mid_channels, self.out_channels)

    def forward(self, x, adj_mat):
        # print(x.size()[0])

        x = F.elu(self.igcn_conv1(x, adj_mat) + self.lin1(x))
        # x = F.elu(self.igcn_conv2(x, adj_mat) + self.lin2(x))
        x = self.igcn_conv3(x, adj_mat) + self.lin3(x)
        return x