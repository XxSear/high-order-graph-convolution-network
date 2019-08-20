#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/10 19:25

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch


def get_first_order_neighbor():
    pass


def edge_index_2_node_vec(edge_index, node_vec_mat):
    # print(edge_index.size(), node_vec_mat.size())
    # edge_index = 2*n    node_vex_mat = n*f_z
    source_node_index = edge_index[0,:].long()
    # print(source_node_index.size())
    target_node_index = edge_index[1,:].long()
    source_node_vec = node_vec_mat[source_node_index]
    target_node_vec = node_vec_mat[target_node_index]
    # source_node_vec = torch.index_select(node_vec_mat, 0, edge_index[0])
    # target_node_vec = torch.index_select(node_vec_mat, 0, edge_index[1])
    # print(source_node_vec.size())
    # print(source_node_vec.mean())
    # print(target_node_vec.mean())
    return torch.cat([source_node_vec, target_node_vec],dim=-1)



if __name__ == '__main__':
    edge = torch.tensor([[1,2,0,1],[0,2,1,1]], dtype=torch.long)
    node = torch.tensor([
        [0.2, 1, 3],
        [0, 0.6, 1],
        [0.3, 2, 1]], dtype=torch.float32)
    res = edge_index_2_node_vec(edge, node)
    print(res)
    print(torch.index_select(node, 0, edge[0]))