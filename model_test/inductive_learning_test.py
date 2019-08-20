#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/25 13:25

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch
import torch.nn.functional as F

# inductive learning 归纳学习
# 数据集中有多个图
# 部分图做训练集，使用图内所有结点的标签

# 该测试类 输入为 模型 和 数据集
# 修改训练集数据集的比例  只能以图为单位

from GNN_Implement.data_loader import PPI
from sklearn import metrics
from random import choice
import numpy as np

class InductiveLearningTest(object):
    def __init__(self, model_, dataset):
        self.model_ = model_
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(3)
        self.dataset = dataset
        # self.dataset = PPI()

        self.dataset.to_device(self.device)

        self.node_ft_size = self.dataset.get_node_ft_size()
        # self.node_num = self.dataset.get_node_size()
        self.label_num = self.dataset.get_label_num()

        self.built_model()

    def built_model(self):
        self.model = self.model_(
            self.node_ft_size,
            self.label_num
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.epochs = 1000
        self.test_acc_list = []
        self.loss_op = torch.nn.BCEWithLogitsLoss()

    def train(self):
        self.model.train()
        total_loss = 0
        train_f1 = 0
        for _ in range(len(self.dataset.train_graphs)):
            one_train_graph = choice(self.dataset.train_graphs)

            self.optimizer.zero_grad()
            pred = self.model(one_train_graph.node_ft , one_train_graph.edge_list).float()
            label = one_train_graph.node_label.float()

            loss = self.loss_op(pred, label)

            pred = (pred > 0).float().cpu()
            label = label.float().cpu()

            micro_f1 = metrics.f1_score(label, pred, average='micro')
            # print('node_num = ', one_train_graph.node_num, '1-pos = ', one_train_graph.node_label.sum(), 'micro_f1 = ', micro_f1)
            train_f1 += micro_f1
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.dataset.train_graphs), train_f1 / len(self.dataset.train_graphs)

    def metrics_model(self, graph_list):
        self.model.eval()
        total_micro_f1 = 0
        total_micro_p = 0
        total_micro_r = 0
        right_num = 0
        all_num = 0
        for one_graph in graph_list:
            with torch.no_grad():
                out = self.model(one_graph.node_ft , one_graph.edge_list)
            pred = (out > 0).float().cpu()
            label = one_graph.node_label.float().cpu()

            micro_f1 = metrics.f1_score(label, pred, average='micro')
            # print('node_num = ', one_graph.node_num, '1-pos = ', one_graph.node_label.sum(), 'micro_f1 = ', micro_f1)
            total_micro_f1 += metrics.f1_score(label, pred, average='micro')
            total_micro_p += metrics.precision_score(label, pred, average='micro')
            total_micro_r += metrics.recall_score(label, pred, average='micro')
        # return float(right_num) / all_num
        return total_micro_f1 / len(graph_list), total_micro_p / len(graph_list), total_micro_r / len(graph_list)

    def start(self, display=True):

        for epoch in range(1, self.epochs+1):
            loss, train_f1 = self.train()

            # train_f1 = self.metrics_model(self.dataset.train_graphs)
            # valid_f1 = self.metrics_model(self.dataset.valid_graphs)
            test_f1, test_p, test_r = self.metrics_model(self.dataset.test_graphs)

            print('epoch = ', epoch, 'loss = ', loss, 'train_f1 = ', train_f1, 'test_f1 = ', test_f1, 'test_p = ',
                  test_p, 'test_r = ', test_r)

            # if display is True:
            #     print('Epoch: {:02d}, Loss: {:.4f}, train_f1: {:.4f}, train_f1: {:.4f}, train_f1: {:.4f}'.format(
            #         epoch, loss, valid_f1, valid_f1, test_f1))



if __name__ == '__main__':
    from GNN_Implement.model.GCN import GCN
    from GNN_Implement.model.modified_model.ppi_GCN import GCN
    from GNN_Implement.model.modified_model.ppi_GAT import GAT
    # from GNN_Implement.model.modified_model.ppi_GCN2 import GCN2
    from GNN_Implement.model.GGNN import GGNN

    demo_test = InductiveLearningTest(GGNN, PPI())
    # demo_test = InductiveLearningTest(GAT, PPI())
    # demo_test = InductiveLearningTest(GCN, PPI())

    # print(demo_test.dataset.train_graphs[0].node_label.device)

    demo_test.start()