#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/3 13:07

base Info
"""
__author__ = '周朋飞'
__version__ = '1.0'
import torch
import numpy as np
from torch.utils import data
from data_loader.process_srcipt.torch_dataset import TorchDataset

# 数据全是torch tensor

class Data(object):
    def __init__(self,
                 torch_dataset_cls=TorchDataset,
                 all_node_feature=None,
                 edge_index=None,
                 edge_attr=None,
                 all_node_label=None,
                 pos=None,
                 adj_table=None,
                 batch_size=1,
                 pre_split_train_index=None,
                 pre_split_valid_index=None,
                 pre_split_test_index=None
                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dataset = torch_dataset_cls

        self.batch_size = batch_size

        self.adj_table = self.to_device(adj_table)
        # self.bias_adj_table
        self.all_node_feature = self.to_device(all_node_feature)
        self.edge_index = self.to_device(edge_index)
        self.edge_attr = self.to_device(edge_attr)
        self.all_node_label = self.to_device(all_node_label)

        self.train_index = self.to_device(pre_split_train_index)
        self.valid_index = self.to_device(pre_split_valid_index)
        self.test_index = self.to_device(pre_split_test_index)

        # self.pos = pos.to(self.device) if pos is not None else pos

        self.shuffled_index = None

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def to_device(self, data):
        if data is not None:
            return data.to(self.device)
        else:
            return data

    def __sizeof__(self):
        return self.all_node_feature.size()[0]

    def train_test_split(self, split_param, mode='ratio', shuffle=True):
        '''
        :param param: The proportion or number of data divided
        :param mode: 'ratio' or 'numerical'
        :return:
        '''
        if mode == 'ratio':
            split_param = np.array(split_param)
            ratio = split_param / np.sum(split_param)
            train_size, valid_size, test_size = map(int, np.ceil(ratio * self.__sizeof__()))

        elif mode == 'numerical':
            split_param = np.array(split_param)
            assert np.sum(split_param) <= self.__sizeof__()
            train_size, valid_size, test_size = split_param

        self.shuffled_index = np.arange(self.__sizeof__(), dtype=np.long)
        if shuffle:
            np.random.shuffle(self.shuffled_index)

        self.train_index = torch.from_numpy(self.shuffled_index[:train_size])
        self.valid_index = torch.from_numpy(self.shuffled_index[train_size: train_size + valid_size])
        self.test_index = torch.from_numpy(self.shuffled_index[train_size + valid_size: train_size + valid_size + test_size])

        # self.train_labels = self.all_node_label[self.train_index]
        # self.valid_labels = self.all_node_label[self.valid_index]
        # self.test_labels = self.all_node_label[self.test_index]

        self.train_set = self.torch_dataset(self.train_index, self.all_node_label, self.all_node_feature, self.adj_table)
        self.valid_set = self.torch_dataset(self.valid_index, self.all_node_label, self.all_node_feature, self.adj_table)
        self.test_set = self.torch_dataset(self.test_index, self.all_node_label, self.all_node_feature, self.adj_table)

        self.train_dataloader = data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.valid_dataloader = data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_dataloader = data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=shuffle)

        # print(self.node_label[1])
        print('bulid dataloader')
        return True

    def overwrite_data(self, **kwargs):
        keys = list(kwargs.keys())






