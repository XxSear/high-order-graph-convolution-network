#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/17 9:31

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

from data_loader import Cora, Citeseer, Pubmed, PPI
from model.GCN import GCN
from model.GAT import GAT
from model_test.graph_test_base import GraphBase
import torch


class GCNTest(GraphBase):
    def __init__(self, dataset):

        # dataset = Cora()
        # dataset = Citeseer()
        # dataset = Pubmed()
        model = GCN
        # dataset.train_test_split([0.01, 0.1, 0.1], shuffle=False)
        super(GCNTest, self).__init__(dataset, model)
        # self.start()

class GATTest(GraphBase):
    def __init__(self, dataset):
        # dataset = Cora()
        # dataset = Citeseer()
        # dataset = Pubmed()
        model = GAT

        super(GATTest, self).__init__(dataset, model)
        # self.dataset.train_test_split([0.8, 0.1, 0.1], shuffle=False)
        self.model = self.model_(
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
        ).to(self.device)

        # self.start()

class GGNNTest(GraphBase):
    def __init__(self, dataset):
        # dataset = Cora()
        # dataset = Citeseer()
        # dataset = PPI()
        model = GAT

        super(GGNNTest, self).__init__(dataset, model)
        self.dataset.train_test_split([0.8, 0.1, 0.1], shuffle=False)
        self.model = self.model_(
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
        ).to(self.device)


def ratio_test(test_model, ratio_list, test_time=3):
    print(test_model.dataset.dataset_name)
    for ratio in ratio_list:
        test_model.dataset.train_test_split(ratio, shuffle=False)
        model_res_list = []

        for _ in range(test_time):
            test_model.built_net()
            res = test_model.start(display_acc=False)
            model_res_list.append(res)

        print(sum(model_res_list) / float(test_time))
        # print(ratio ,' ', sum(model_res_list) / float(test_time))

if __name__ == '__main__':
    torch.cuda.set_device(1)
    ratio_list = [
        # [0.005, 0.1, 0.1],
        [0.01, 0.1, 0.1],
        [0.02, 0.1, 0.1],
        [0.03, 0.1, 0.1],
        [0.05, 0.1, 0.1],
        [0.07, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
        [0.3, 0.1, 0.1],
        [0.4, 0.1, 0.1],
        [0.5, 0.1, 0.1],
        [0.6, 0.1, 0.1],
        [0.7, 0.1, 0.1],
        [0.8, 0.1, 0.1],
    ]

    # dataset = Cora()
    # dataset = Citeseer
    # dataset = Pubmed()
    for dataset in [Cora, Citeseer, Pubmed]:
        # net = GCNTest(dataset())
        net = GATTest(dataset())
        ratio_test(net, ratio_list)


    # GATTest(Pubmed())
