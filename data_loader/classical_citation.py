#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/7 21:14

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

from data_loader.process_srcipt.classical_citation_porcess_file import ClassicalCitationPorcess
from data_loader.process_srcipt.data import Data
from data_loader.process_srcipt.torch_dataset import TorchDataset
from model_component.utils.adj_mat import adj_2_bias
from model_component.utils.add_self_loop import list_add_self_loops
from data_loader.process_srcipt.process_tools import normalize

import numpy as np
import torch

class ClassicalCitation(object):
    def __init__(self, dataset_name, normalize_ft=True):
        # loading data
        # print('ClassicalCitation')
        self.dataset_name = dataset_name
        data_dict = ClassicalCitationPorcess(dataset_name).processed_data
        self.all_node_feature = data_dict['x']
        if normalize_ft is True:
            self.all_node_feature = normalize(self.all_node_feature)
        self.all_node_label = data_dict['y']
        self.edge_index = data_dict['edge_index']
        self.adj_table = data_dict['adj_table']
        self.pre_split_train_index = data_dict['train_index']
        self.pre_split_valid_index = data_dict['valid_index']
        self.pre_split_test_index = data_dict['test_index']
        self.data_loader = Data

        self.node_feature_size = self.all_node_feature.size()[-1]
        self.node_size = self.all_node_label.size()[0]
        self.label_size = len(set(list(self.all_node_label.numpy())))

        self.bias_adj_mat = adj_2_bias(self.adj_table)

        self.train_index = self.pre_split_train_index
        self.valid_index = self.pre_split_valid_index
        self.test_index = self.pre_split_test_index
        self.add_self_loop_edge_list = list_add_self_loops(self.node_size, self.edge_index)

        # self.add_self_loop_mat_adj = add_self_loops(self.node_size, self.adj_table)
        # self.bias_adj_mat = adj_2_bias(self.add_self_loop_mat_adj)



    # if the number of node's neighbor is different, batch_size must be 1
    def get_data_loader(self, split_param, mode='ratio', shuffle=True, batch_size=1, torch_dataset_cls=TorchDataset):
        self.data_loader = Data(all_node_feature=self.all_node_feature,
                                 edge_index=self.edge_index,
                                 all_node_label=self.all_node_label,
                                 adj_table=self.adj_table,
                                 batch_size=batch_size,
                                 torch_dataset_cls=torch_dataset_cls
                                 )
        self.data_loader.train_test_split(split_param, mode=mode, shuffle=shuffle)

        return self.data_loader.train_dataloader, \
               self.data_loader.valid_dataloader, \
               self.data_loader.test_dataloader

    def to_device(self, device):
        self.all_node_feature = self.all_node_feature.to(device)
        self.all_node_label = self.all_node_label.to(device)
        self.edge_index = self.edge_index.to(device)
        self.adj_table = self.adj_table.to(device)
        self.pre_split_train_index = self.pre_split_train_index.to(device)
        self.pre_split_valid_index = self.pre_split_valid_index.to(device)
        self.pre_split_test_index = self.pre_split_test_index.to(device)
        # self.bias_adj_mat = self.bias_adj_mat.to(device)
        self.add_self_loop_edge_list = self.add_self_loop_edge_list.to(device)

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
            # ratio = split_param / np.sum(split_param)

            train_size, valid_size, test_size = map(int, np.ceil(split_param * self.node_size))

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

    def get_n_order_adj_list(self, order_num, fliter_path_num=1, device='cpu'):
        n_order_adj_mat = self.adj_table
        for i in range(order_num):
            n_order_adj_mat = torch.mm(n_order_adj_mat, self.adj_table)

        n_order_adj_list = (n_order_adj_mat > fliter_path_num).nonzero()  # 不加入self-loop 过滤邻居
        n_order_adj_list = torch.transpose(n_order_adj_list, 0, 1)
        n_order_adj_list = n_order_adj_list.to(device)
        return n_order_adj_list

if __name__ == '__main__':
    demo = ClassicalCitation('Cora')
    print(demo.test_index)
