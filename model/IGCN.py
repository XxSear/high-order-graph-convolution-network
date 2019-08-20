#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/23 21:07

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch.nn.functional as F

import torch
import torch.nn as nn

from model_component.conv.igcn_conv_input_mat import IGCNConv


class IGCN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 k=5,
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
        self.mid_channels = in_channels

        self.igcn_conv1 = IGCNConv(self.in_channels, self.mid_channels,
                                   bias=self.bias, k=self.k, type=self.model_type, alpha=self.alpha)
        self.igcn_conv2 = IGCNConv(self.mid_channels,  self.out_channels,
                                   bias=self.bias, k=self.k, type=self.model_type, alpha=self.alpha)

    def forward(self, node_ft, adj_mat):
        node_feature = F.dropout(node_ft,
                                 p=self.dropout,
                                 training=self.training)

        node_state = self.igcn_conv1(node_feature, adj_mat)

        node_state = F.elu(node_state)
        node_state = F.dropout(node_state,
                               p=self.dropout,
                               training=self.training)
        out = self.igcn_conv2(node_state, adj_mat)

        return F.log_softmax(out, dim=1)
        # return out

class IGCN_AR(nn.Module):
    def __init__(self, k):
        super(IGCN_AR, self).__init__()

