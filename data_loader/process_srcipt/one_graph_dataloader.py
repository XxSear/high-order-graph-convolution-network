#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/25 10:45

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import numpy as np
import torch


class OneGraph(object):
    def __init__(self, normalize_ft=True):

        self.node_ft_size = None
        self.label_size = None
        self.node_num = None

        self.node_label = None
        self.node_ft = None

        self.adj_table = None
        self.edge_list = None
        self.edge_ft = None

        self.pre_split_train_index = None
        self.pre_split_valid_index = None
        self.pre_split_test_index = None


    def to_device(self, device, attr_name_list):
        for attr_name in attr_name_list:
            graph_data = getattr(self, attr_name)
            graph_data = graph_data.to(device)
            # print(graph_data.device)
            setattr(self, attr_name, graph_data)
            # print(getattr(self, attr_name).device)

        self.node_num = self.node_ft.size()[0]

        # self.node_label = self.node_label.to(device)
        # self.node_ft = self.node_ft.to(device)
        # self.adj_table = self.adj_table.to(device)
        # self.edge_list = self.edge_list.to(device)

        # self.pre_split_train_index = self.pre_split_train_index.to(device)
        # self.pre_split_valid_index = self.pre_split_valid_index.to(device)
        # self.pre_split_test_index = self.pre_split_test_index.to(device)

        # self.bias_adj_mat = self.bias_adj_mat.to(device)
        # self.add_self_loop_edge_list = self.add_self_loop_edge_list.to(device)

    def train_test_split(self, split_param, mode='ratio', shuffle=False):
        '''
        :param param: The proportion or number of data divided
        :param mode: 'ratio' or 'numerical'
        :return:
        '''
        # seed = 1
        # torch.manual_seed(seed)  # 为CPU设置随机种子
        # torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        # torch.cuda.manual_seed_all(seed)

        if mode == 'ratio' and sum(split_param) <= 1:
            split_param = np.array(split_param)
            ratio = split_param / np.sum(split_param)
            train_size, valid_size, test_size = map(int, np.ceil(ratio * self.node_size))

        elif mode == 'numerical' or sum(split_param) > 1:
            split_param = np.array(split_param)
            # print(np.sum(split_param), self.node_size)
            assert np.sum(split_param) <= self.node_size
            train_size, valid_size, test_size = split_param

        if shuffle is True:
            node_index = torch.randperm(self.node_size)
        else:
            node_index = torch.arange(self.node_size)

        self.train_index = node_index[0:train_size]
        self.valid_index = node_index[train_size: valid_size + train_size]
        # self.test_index = node_index[valid_size + train_size: valid_size + train_size + test_size]
        self.test_index = node_index[-test_size:]

        return True

if __name__ == '__main__':
    graph = OneGraph()
    print(graph.__dict__)


