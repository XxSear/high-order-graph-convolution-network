#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/8/1 0:54

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch.nn.functional as F

import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.6,
                 layer=2,
                 mid_layer_channel=64,
                 ):
        super(MLP, self).__init__()
        self.mid_layer_channel = mid_layer_channel
        self.dropout = dropout
        self.num_layer = layer
        self.layer = layer
        self.liner_lyaer_list = nn.ModuleList()
        for idx in range(layer):
            in_size = mid_layer_channel
            out_size = mid_layer_channel
            if idx == 0:
                in_size = in_channels
            if idx == layer - 1:
                out_size = out_channels

            liner_layer = torch.nn.Linear(in_size, out_size)
            self.liner_lyaer_list.append(liner_layer)

    def forward(self, node_tf):

        for liner_layer in self.liner_lyaer_list:
            # node_tf = F.dropout(node_tf,
            #                     p=self.dropout,
            #                     training=self.training)
            node_tf = F.elu(liner_layer(node_tf))

        return node_tf
        # return F.log_softmax(node_tf, dim=-1)
