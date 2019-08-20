#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/2 14:18

base Info
"""
__author__ = '周朋飞'
__version__ = '1.0'

from data_loader.process_srcipt.torch_dataset import TorchDataset
from data_loader.classical_citation import ClassicalCitation

class my_dataset(TorchDataset):
    def __init__(self, node_index, node_label, all_node_feature, adj_table, all_edge_feature=None, edge_index=None,
                 **kwargs):
        super(my_dataset, self).__init__(node_index, node_label, all_node_feature, adj_table,
                                         all_edge_feature=all_edge_feature, edge_index=edge_index ,**kwargs)

    def __getitem__(self, index):
        # example
        print('over')
        return 1,2,3


class Cora(ClassicalCitation):
    def __init__(self):
        super(Cora, self).__init__('Cora')

    @staticmethod
    def get_dataset():
        return Cora()


if __name__ == '__main__':
    # dataset = ClassicalCitation('Pubmed')
    # dataset = Cora()
    # train_data_loader, valid_data_loader, test_data_loader = dataset.get_data_loader([6,2,2], torch_dataset_cls=my_dataset)
    # train_data_loader, valid_data_loader, test_data_loader = dataset.get_data_loader([6,2,2])
    #
    # for batch_data in train_data_loader:
    #     center_node_feature, center_node_label, neighbor_nodes_feature = batch_data
    #     print(center_node_feature, center_node_label, neighbor_nodes_feature)
    #     break
    # print(len(dataset.pre_split_test_index))
    # print(len(dataset.pre_split_train_index))
    # print(dataset.all_node_label.size())
    # print(dataset.label_size)
    # print(type(dataset.adj_table))
    # print(dataset.adj_table)
    # print(dataset.edge_index)
    # print(dataset.all_node_feature.device)
    # print(dataset.adj_table)
    # dataset.train_test_split([0.4, 0.4, 0.2])

    demo = Cora.get_dataset()
    print(demo.edge_index.size())
    print(demo.add_self_loop_edge_list.size())
