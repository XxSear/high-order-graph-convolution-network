#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/15 16:42

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


class GCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 dropout=0.6,
                 bias=True
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias is True:
            self.bias = Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, nodes_ft, adj_list):

        num_nodes = nodes_ft.size()[0]
        node_hidden_state = torch.mm(nodes_ft, self.weight)
        # 这里加一个dropout貌似效果好一点
        F.dropout(node_hidden_state, p=self.dropout, training=self.training)
        adj_list, alpha = self.norm(adj_list, num_nodes, None, improved=False)
        # print('alpha = ', alpha)
        # print('node_hidden_state = ', node_hidden_state.size(), node_hidden_state)
        # print('adj_list = ', adj_list)
        vals = torch.index_select(node_hidden_state, 0, adj_list[1]) * alpha.view(-1, 1)
        # print('vals = ',vals)
        vals = scatter_('add', vals, adj_list[0], dim_size=nodes_ft.size()[0])

        if self.bias is not None:
            vals = vals + self.bias

        # print('out_node_state size = ', vals.size())
        return vals


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index = list_add_self_loops(num_nodes, edge_index)
        loop_weight = torch.full(
            (num_nodes, ),
            1 if not improved else 2,
            dtype=edge_weight.dtype,
            device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


if __name__ == '__main__':
    dataset = Cora()
    net = GCNConv(
        dataset.node_feature_size,
        dataset.label_size,
    )
    net(dataset.all_node_feature, dataset.edge_index)