#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/8 20:33

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

from data_loader.Cora import Cora
from process_srcipt.torch_dataset import TorchDataset

def test_label():
    dataset = Cora()
    train_data_loader, valid_data_loader, test_data_loader = dataset.get_data_loader([6, 2, 2])
    data_label = dataset.data_loader.all_node_label
    ori_data_label = dataset.all_node_label
    cal_data_label = dataset.data_loader.all_node_label
    final_data_label = dataset.data_loader.train_set.all_node_label
    print(ori_data_label)
    print(cal_data_label)
    print(final_data_label)


    for i, batch_data in enumerate(test_data_loader, 0):
        center_node_index, center_node_feature, center_node_label, neighbor_nodes_feature = batch_data
        center_node_index.cpu()
        correct_label = data_label[center_node_index]
        print("index = ",center_node_index,  correct_label,'cal_label = ', center_node_label,'ori = ', ori_data_label[center_node_index])
        break

if __name__ == '__main__':
    test_label()