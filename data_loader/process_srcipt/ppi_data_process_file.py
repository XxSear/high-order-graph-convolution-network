#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/25 10:20

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import os.path as osp
import os
import numpy as np
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

from itertools import product
from itertools import chain
import networkx as nx
from networkx.readwrite import json_graph
import json
import torch

from data_loader.process_srcipt.base_process import BaseProcess
from data_loader.process_srcipt.one_graph_dataloader import OneGraph
from data_loader.process_srcipt.process_tools import remove_self_loops

class PPIProcess(BaseProcess):
    def __init__(self, reprocess=False):

        self.dataset_name = 'ppi'
        self.dataset_dir = os.path.join(os.path.abspath(osp.dirname(__file__)), '..', 'dataset', self.dataset_name)

        self.train_graphs = []
        self.valid_graphs = []
        self.test_graphs = []

        self.all_graph = chain(self.train_graphs, self.valid_graphs, self.test_graphs)

        super(PPIProcess, self).__init__(self.dataset_name, self.dataset_dir, reprocess=reprocess)


    # from https://github.com/rusty1s/pytorch_geometric
    # @property
    # def _processed_file_list(self):
    #     splits = ['train', 'valid', 'test']
    #     files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
    #     return ['{}_{}'.format(s, f) for s, f in product(splits, files)]

    def _process(self):
        for s, split in enumerate(['train', 'valid', 'test']):
            path = osp.join(self.raw_dir, '{}_graph.json').format(split)
            with open(path, 'r') as f:
                G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

            x = np.load(osp.join(self.raw_dir, '{}_feats.npy').format(split))
            x = torch.from_numpy(x).to(torch.float)

            y = np.load(osp.join(self.raw_dir, '{}_labels.npy').format(split))
            y = torch.from_numpy(y).to(torch.float)

            # 图拆分
            path = osp.join(self.raw_dir, '{}_graph_id.npy').format(split)
            idx = torch.from_numpy(np.load(path)).to(torch.long)
            idx = idx - idx.min()
            print(idx)
            for i in range(idx.max().item() + 1):
                mask = idx == i

                G_s = G.subgraph(mask.nonzero().view(-1).tolist())
                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                edge_index = edge_index - edge_index.min()
                # 保留 图中的 self-loop
                edge_index, _ = remove_self_loops(edge_index)


                node_ft = x[mask]
                node_label = y[mask]

                # print(s, split, 'i = ', i, edge_index )
                # print('node_ft size = ', node_ft.size(), 'node_label size = ', node_label.size())
                one_graph = OneGraph()
                one_graph.node_ft = node_ft
                one_graph.node_label = node_label
                one_graph.edge_list = edge_index

                graph_list = getattr(self, split+'_graphs')
                graph_list.append(one_graph)
                print('split :', len(graph_list), 'node_num = ', node_label.size(), ' 1-pos = ', node_label.sum())
                # setattr(self, split+'_graphs', graph_list)

        self._save_processed_file()
        return True


    def get_node_ft_size(self):
        return self.train_graphs[0].node_ft.size()[1]

    def get_node_num(self):
        return self.train_graphs[0].node_ft.size()[0]

    def get_label_num(self):
        return self.train_graphs[0].node_label.size()[1]

    def to_device(self, device, select_att=None):
        if select_att is None:
            attr_name_list = ['node_label', 'node_ft', 'edge_list']
        else:
            attr_name_list = select_att
        for one_graph in chain(self.train_graphs, self.valid_graphs, self.test_graphs):
            one_graph.to_device(device, attr_name_list)

if __name__ == '__main__':
    ppi = PPIProcess(reprocess=True)


    # test_graph_id = np.load(ppi.dataset_dir + '/' + 'graph_id' + '/' + 'test_graph_id.npy')
    # print(test_graph_id.shape)
    # test_graph_labels = np.load(ppi.dataset_dir + '/' + 'graph_id' + '/' + 'test_labels.npy')
    # print(test_graph_labels.shape, test_graph_labels[0])