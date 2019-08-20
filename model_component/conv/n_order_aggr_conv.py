#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/6/17 19:46

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F


class NOrderAggrConv(torch.nn.Module):
    def __init__(self, in_channels, n=2, depth=2, dropout=0.6, trans=True):
        super(NOrderAggrConv, self).__init__()
        self.dropout = 0.5
        self.in_channels = in_channels
        self.trans = trans
        self.n = n
        self.depth = depth
        self.dropout = dropout

        self.combine_weight_layers = torch.nn.ParameterList()
        for i in range(self.depth - 1):
            tmp_layer = Parameter(
                    torch.Tensor(1, in_channels, self.n, self.n))
            init.xavier_normal_(tmp_layer)
            self.combine_weight_layers.append(tmp_layer)

        tmp_layer = Parameter(
            torch.Tensor(1, in_channels, self.n, 1))
        init.xavier_normal_(tmp_layer)
        self.combine_weight_layers.append(tmp_layer)



    def forward(self, vecs):
        # vecs  [n*k,....]
        # print(type(vecs), vecs)
        tuple_vecs = tuple(vec for vec in vecs)
        # print(type(tuple_vecs))
        vecs = torch.cat(tuple_vecs, -1)
        node_num = vecs.size()[0]

        # vecs = F.dropout(vecs,
        #                  p=self.dropout,
        #                  training=self.training)

        for combine_weight_layer in self.combine_weight_layers:
            combine_state = vecs.view(node_num, self.in_channels, self.n, 1)
            # print(type(combine_state), combine_state.device,
            #       type(combine_weight_layer), combine_weight_layer.device)

            # print('combine_state_ size', combine_state.size())
            combine_state = combine_state * combine_weight_layer
            # print('combine_state  size', combine_state.size())

            combine_state = torch.sum(combine_state, 2)  # num_node * in_channels * n
            combine_state = F.leaky_relu(combine_state, negative_slope=0.3)

        combine_state = torch.sum(combine_state, -1) # num_node * in_channels
        # combine_state = torch.max(combine_state, dim=-1)[0]

        return combine_state


if __name__ == '__main__':
    model = NOrderAggrConv(5, depth=3)

