#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/24 12:51

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from GNN_Implement.model_component.utils.adj_mat import get_laplace_mat


class IGCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 dropout=0.6,
                 bias=True,
                 type='rnm',
                 k=5,
                 alpha=10
                 ):
        super(IGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        assert type in ['rnm', 'rw', 'ap']
        self.type = type
        '''
            rnm: Symmetric normalized Laplacian 
            rw: Random walk normalized Laplacian
            ap:  Auto-Regressive (AR) filter   
        '''
        self.k = k
        self.alpha = alpha

        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias is True:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, node_ft, adj_mat):
        node_state = torch.mm(node_ft, self.weight)

        if self.type == 'rnm':
            laplace_mat = get_laplace_mat(adj_mat, type='sym')
            node_state = self.rnm_filter(node_state, laplace_mat, k=self.k)

        elif self.type == 'rw':
            laplace_mat = get_laplace_mat(adj_mat, type='sym')
            node_state = self.rw_filter(node_state, laplace_mat, k=self.k)

        elif self.type == 'ap':
            laplace_mat = get_laplace_mat(adj_mat, type='sym')
            # print('bf node_state = ', node_state)
            node_state = self.ap_approximate_filter(node_state, laplace_mat, alpha=self.alpha)
            # print('ap node_state = ', node_state)

        if self.bias is not None:
            node_state = node_state + self.bias

        return node_state

    @staticmethod
    def rnm_filter(node_state, adj_mat, k=5):
        new_ft = node_state
        for _ in range(k):
            new_ft = torch.mm(adj_mat, new_ft)
        return new_ft

    @staticmethod
    def rw_filter(node_state, adj_mat, k=10):
        new_ft = node_state
        for _ in range(k):
            new_ft = (torch.mm(adj_mat, new_ft) + new_ft) / 2
        return new_ft

    @staticmethod
    def ap_approximate_filter(node_state, adj_mat, alpha=10):
        alpha = 1.0 / alpha
        k = math.ceil(4 / alpha)

        tmp_adj_mat = adj_mat / (1+alpha)
        new_ft = node_state
        for _ in range(4):
            new_ft = torch.mm(tmp_adj_mat, new_ft)
            new_ft += node_state
        new_ft *= alpha / (alpha + 1)
        return new_ft
