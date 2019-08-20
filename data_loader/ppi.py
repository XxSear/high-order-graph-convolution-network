#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/21 12:58

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
import os
import json
import numpy as np
import torch
from data_loader.process_srcipt.base_dataset import Dataset
from data_loader.process_srcipt.ppi_data_process_file import PPIProcess


class PPI(PPIProcess):
    def __init__(self):
        super(PPI, self).__init__()




if __name__ == '__main__':
    demo = PPI()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demo.to_device(device)

    # print('train')
    # for graph in demo.train_graphs:
    #     print('graph.node = ', graph.node_num, 'true_num = ', graph.node_label.sum())
    #
    #
    # print('test')
    # for graph in demo.test_graphs:
    #     print('graph.node = ', graph.node_num, 'true_num = ', graph.node_label.sum())
