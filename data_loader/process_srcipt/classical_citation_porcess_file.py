#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/4 10:52

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import os.path as osp
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
from torch_sparse import coalesce

import torch
from itertools import repeat

from data_loader.process_srcipt.base_dataset import Dataset
from data_loader.process_srcipt.base_dataset import processed_dir_name
from data_loader.process_srcipt.base_dataset import dataset_obj_file_suffix
from data_loader.process_srcipt.process_tools import remove_self_loops
from data_loader.process_srcipt.process_tools import read_txt_array
from data_loader.process_srcipt.data import Data
from data_loader.process_srcipt.process_tools import adj_dict_2_torch

# 该模块负责原始数据的读取
# 将数据转换成固定的格式
# x_f, y_f, edge_f, adj_table

class ClassicalCitationPorcess(Dataset):
    def __init__(self, dataset_name, reprocess=True):
        self.dataset_name = dataset_name
        assert self.dataset_name in ["Cora", "Citeseer", "Pubmed"]
        self.reprocess = reprocess
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'dataset', dataset_name)
        Dataset.__init__(self, self.path, self.reprocess)


    # 处理过的文件列表
    def _processed_file_list(self):
        # file_name = self.dataset+'.dat'
        names = [osp.join(self.path, processed_dir_name , self.dataset_name+dataset_obj_file_suffix)]
        return names

    def _process(self):
        '''
        copy from https://github.com/rusty1s/pytorch_geometric
        '''
        raw_file_names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        items = [self._read_file(self.raw_dir, self.dataset_name, tag) for tag in raw_file_names]
        x, tx, allx, y, ty, ally, graph, test_index = items
        # print('test_index = ',sorted( test_index.numpy()))
        train_index = torch.arange(y.size(0), dtype=torch.long)
        val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
        sorted_test_index = test_index.sort()[0]
        if self.dataset_name.lower() == 'citeseer':
            # There are some isolated nodes in the Citeseer graph, resulting in
            # none consecutive test indices. We need to identify them and add them
            # as zero vectors to `tx` and `ty`.
            len_test_indices = (test_index.max() - test_index.min()).item() + 1

            tx_ext = torch.zeros(len_test_indices, tx.size(1))
            tx_ext[sorted_test_index - test_index.min(), :] = tx
            ty_ext = torch.zeros(len_test_indices, ty.size(1))
            ty_ext[sorted_test_index - test_index.min(), :] = ty

            tx, ty = tx_ext, ty_ext

        x = torch.cat([allx, tx], dim=0)        # all x
        y = torch.cat([ally, ty], dim=0).max(dim=1)[1]  # all y

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        # print(x.size(), y.size())
        edge_index = self.edge_index_from_dict(graph, num_nodes=y.size(0))
        # print('buliding data obj')
        # self.data = Data(x=x, y=y, edge_index=edge_index, adj_table=graph, batch_size=2)

        #graph collections to torch
        graph = adj_dict_2_torch(graph)

        self.processed_data = {'x':x, 'y':y, 'edge_index':edge_index,
                               'adj_table':graph, 'train_index':train_index,
                               'valid_index':val_index, 'test_index':sorted_test_index}
        self._save_processed_file()

    # load pickle file
    def _read_file(self, folder, prefix, name):
        path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))
        if name == 'test.index':
            return read_txt_array(path, dtype=torch.long)
        with open(path, 'rb') as f:
            if sys.version_info > (3, 0):
                out = pickle.load(f, encoding='latin1')
            else:
                out = pickle.load(f)

        if name == 'graph':
            return out

        out = out.todense() if hasattr(out, 'todense') else out
        out = torch.Tensor(out)
        return out

    # 将所有的边提取出来，合并相同边，放在一个tensor里  [s1, s2 ...][t1, t2 ...]
    def edge_index_from_dict(self, graph_dict, num_nodes=None):
        row, col = [], []
        for key, value in graph_dict.items():
            row += repeat(key, len(value))
            col += value
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)

        # NOTE: There are duplicated edges and self loops in the datasets. Other
        # implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)  # 合并相同的边，并加权
        return edge_index

    def _save_processed_file(self):
        file_names = self._processed_file_list()
        for file_name in file_names:
            path = osp.join(self.path, processed_dir_name, file_name)
            if dataset_obj_file_suffix in file_name:
                with open(path, 'wb') as f:
                    if sys.version_info > (3, 0):
                        pickle.dump(self.__dict__, f)
                    else:
                        pickle.dump(self.__dict__, f)