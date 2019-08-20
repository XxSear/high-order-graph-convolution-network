#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/21 13:11

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import os
import json
import numpy as np
import torch
from networkx.readwrite import json_graph
import networkx as nx
from model_component.utils.add_self_loop import list_add_self_loops
# 将前人处理的数据读入
# 对于PPI将其划分成23个子图


class SNAP(object):
    def __init__(self, dataset_name):
        self.data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'dataset', dataset_name, 'raw')
        self.dataset_name = dataset_name.lower()
        # print(os.listdir(self.data_dir))
        self.labels = None
        self.pre_split_train_index = []
        self.pre_split_valid_index = []
        self.pre_split_test_index = []
        self.adj_list = []
        self.all_node_feature = None

        self.load_files()
        self.node_feature_size = self.all_node_feature.size()[1]
        self.label_size = self.all_node_label.size()[1]
        self.node_size = self.all_node_feature.size()[0]
        self.add_self_loop_edge_list = list_add_self_loops(self.node_size, self.edge_index)

        self.train_index = self.pre_split_train_index
        self.valid_index = self.pre_split_valid_index
        self.test_index = self.pre_split_test_index

    def load_files(self):
        graph_json_path = self.data_dir + '/' + self.dataset_name + '-G.json'
        label_json_path = self.data_dir + '/' + self.dataset_name + '-class_map.json'
        node_ft_np_path = self.data_dir + '/' + self.dataset_name + '-feats.npy'
        adj_list_txt_path = self.data_dir + '/' + self.dataset_name + '-walks.txt'

        G = json_graph.node_link_graph(json.load(open(graph_json_path)))

        for n in G.nodes():
            if G.node[n]['test']:
                self.pre_split_test_index.append(n)   # ppi 51420:56943 = 5524
            elif G.node[n]['val']:
                self.pre_split_valid_index.append(n)   # ppi 44906:51419 = 6614
            else:
                self.pre_split_train_index.append(n)  # ppi 0:44905


        with open(label_json_path, 'r') as load_f:
            self.labels = list((json.load(load_f)).values())  # ppi 24 graphs  56944 * 121
        # print(len(labels), len( labels[1] ))

        self.all_node_feature = np.load(node_ft_np_path)  #  ppi (56944, 50)
        # print(node_ft.shape)

        with open(adj_list_txt_path, 'r') as txt_f:
            for edge_str in txt_f.readlines():
                if len(edge_str) > 2:
                    nodes = edge_str.split('\n')[0].split('\t')
                    # print(nodes)
                    self.adj_list.append([int(nodes[0]), int(nodes[1])])
                # break
            # print(self.adj_list)

        # to tensor
        self.all_node_feature = (torch.from_numpy(self.all_node_feature)).float()
        self.all_node_label = torch.Tensor(self.labels).float()
        self.edge_index = torch.Tensor(self.adj_list).long()
        self.edge_index = torch.transpose(self.edge_index, 1, 0)

        self.pre_split_train_index = torch.tensor(self.pre_split_train_index,  dtype=torch.long)
        self.pre_split_valid_index = torch.tensor(self.pre_split_valid_index,  dtype=torch.long)
        self.pre_split_test_index = torch.tensor(self.pre_split_test_index,  dtype=torch.long)

    def to_device(self, device):
        self.all_node_feature = self.all_node_feature.to(device)
        self.all_node_label = self.all_node_label.to(device)
        # self.edge_index = self.edge_index.to(device)
        # self.adj_table = self.adj_table.to(device)
        self.pre_split_train_index = self.pre_split_train_index.to(device)
        self.pre_split_valid_index = self.pre_split_valid_index.to(device)
        self.pre_split_test_index = self.pre_split_test_index.to(device)
        # self.bias_adj_mat = self.bias_adj_mat.to(device)
        self.add_self_loop_edge_list = self.add_self_loop_edge_list.to(device)