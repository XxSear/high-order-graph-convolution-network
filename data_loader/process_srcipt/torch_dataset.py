#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/7 21:04

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
from torch.utils import data


'''
    继承自data.Dataset类，用于构建dataloader
    __getitem__方法中 定义了每一次迭代返回的数据格式
'''
class TorchDataset(data.Dataset):
    def __init__(self, node_index, all_node_label, all_node_feature, adj_table, all_edge_feature=None, edge_index=None, **kwargs):
        self.node_index = node_index
        self.all_node_label = all_node_label
        self.all_node_feature = all_node_feature  # 构建邻居的feature
        self.adj_table = adj_table # dict
        self.edge_index = edge_index # n*2  [[s_index, t_index],]
        self.all_edge_feature = all_edge_feature # None or tensor n_edges * edge_feature


        # print(type(adj_table))
        # print(adj_table.keys())

    def __len__(self):
        return len(self.node_index)

    def __getitem__(self, index):
        # 需要根据任务重写
        center_node = self.node_index[index]
        center_node_index = int(center_node.numpy())
        # print(self.all_node_label)
        # print('center_node_index = ',center_node_index, 'label = ',self.all_node_label[center_node_index]  )

        center_node_label = self.all_node_label[center_node_index]  # 1 * out_feature_size
        center_node_feature = self.all_node_feature[center_node_index] # 1 * inp_feature_size

        neighbor_nodes = self.adj_table[center_node_index]  # return list[node_id, ]
        neighbor_nodes_feature = self.all_node_feature[torch.tensor(neighbor_nodes)]  # n_neighbor * feature_size

        # if self.edge_index  and self.all_edge_feature :
        #     center_node_edges_index = torch.eq(self.edge_index[:,0], 1).nonzero().view(-1)
        #     center_node_edges_feature = self.all_edge_feature[center_node_edges_index]
        #     return center_node_feature, center_node_label, neighbor_nodes_feature, center_node_edges_feature
        # else:
        return center_node_index,center_node_feature, center_node_label, neighbor_nodes_feature