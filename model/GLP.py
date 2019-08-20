#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/24 18:21

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch.nn.functional as F

import torch
import torch.nn as nn


class GLP(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 k=5,
                 alpha=10,
                 model_type='rnm',
                 imporve=False,
                 dropout=0.5,
                 bias=True):
        super(GLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.alpha=alpha
        self.model_type = model_type
        self.dropout = dropout
        self.bias = bias
        self.mid_channels = in_channels

    def forward(self, *input):
        pass



