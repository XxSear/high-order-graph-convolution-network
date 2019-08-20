#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/14 23:32

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F


class AggregatioAttnConv(torch.nn.Module):
    def __init__(self, input_channel, out_channel, head=1, dropout=0.6):
        super(AggregatioAttnConv, self).__init__()
        self.head = head

        self.weight = Parameter(
            torch.Tensor(input_channel, out_channel))

        self.attn_vec = Parameter(
            torch.Tensor(self.head, out_channel))
        self.bias = Parameter(torch.Tensor(out_channel))
        self.out_channel = out_channel
        self.input_channel = input_channel

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        init.xavier_normal_(self.attn_vec)
        # init.xavier_normal_(self.bias)
        init.zeros_(self.bias)


    def forward(self, embed_list):
        n_node = embed_list.size()[1]
        n_graph = embed_list.size()[0]
        # embed_list [n_graph, n_node, self.attn_out_size * self.node_level_attn_head]
        # print('embed_list = ',embed_list.size())

        torch.transpose(embed_list, 0, 1)  # n_node, n_graph, hidden_size
        embed_list = embed_list.view(-1, self.input_channel) # 转化成2维矩阵，用于矩阵torch.mm 二维矩阵乘法
        # print(embed_list.size(), self.weight.size())

        x = torch.mm(embed_list, self.weight).view(
            n_node, n_graph, self.out_channel)

        w = (self.attn_vec * torch.tanh(x)).sum(dim=-1) # [n_node, n_graph]
        # print(w)
        # e = F.leaky_relu(e, negative_slope=0.2)
        beta = F.softmax(w, dim=1) # [n_node, n_graph]

        out = x * (beta.view(n_node, n_graph, 1))  # 先广播 再 元素相乘
        out = torch.transpose(out, 1, 2).sum(-1)
        # n_node, n_graph, self.out_channel -> n_node, self.out_channel, n_graph -> n_node, self.out_channel
        return out, beta


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_list = torch.tensor([
        [[1,2,3,4,5,6],[2,3,4,5,6,7],[2,3,4,5,6,7]],
        [[6,5,4,3,2,1],[7,6,5,4,3,2],[8,7,6,5,4,3]]
    ], dtype=torch.float32).to(device)
    net = AggregatioAttnConv(6, 10).to(device)
    net(embed_list)
