#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/8/1 15:09

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
from GNN_Implement.data_loader import PPI
from sklearn import metrics
from random import choice
import numpy as np
import torch
from itertools import chain
from GNN_Implement.model_component.utils.adj_mat import adj_list_to_n_order_adj_list
from GNN_Implement.model_component.utils.add_self_loop import adj_mat_add_self_loop
from GNN_Implement.model_component.utils.adj_mat import edge_list_2_adj_mat
from GNN_Implement.model_component.utils.adj_mat import adj_mat_to_n_order_adj_mat


class NOrderInductiveLearningTestMat(object):
    def __init__(self, model_, dataset, order=2):
        self.model_ = model_
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        # self.dataset = PPI()
        assert order >= 1
        self.adj_order = order
        self.dataset.to_device(self.device, select_att=['node_label', 'node_ft'])

        self.node_ft_size = self.dataset.get_node_ft_size()
        # self.node_num = self.dataset.get_node_size()
        self.label_num = self.dataset.get_label_num()
        self.get_adj_mat()
        self.built_model()


    # 图太多 放到CPU中
    def get_adj_mat(self, device='cpu'):
        print('bulit adj mat')
        self.train_n_order_adj_mat = [[edge_list_2_adj_mat(
            graph.edge_list, node_num=graph.node_num, device=device)]
            for graph in self.dataset.train_graphs]

        self.valid_n_order_adj_mat = [[edge_list_2_adj_mat(
            graph.edge_list, node_num=graph.node_num, device=device)]
            for graph in self.dataset.valid_graphs]

        self.test_n_order_adj_mat = [[edge_list_2_adj_mat(
            graph.edge_list, node_num=graph.node_num, device=device)]
            for graph in self.dataset.test_graphs]



        for one_graph_list in chain(self.train_n_order_adj_mat,
                                    self.valid_n_order_adj_mat,
                                    self.test_n_order_adj_mat):
            for order_idx in range(1, self.adj_order):
                high_order_adj_mat = adj_mat_to_n_order_adj_mat(one_graph_list[0], order_idx, device=device, fliter_path_num=1)
                one_graph_list.append(high_order_adj_mat)
                print('one_graph_list[0] size = ', one_graph_list[0].size(), 'high_order_adj_list size = ', high_order_adj_mat.size())
        #
        #     # 加入 self--loop
        #     # print(one_graph_list[0].size())
            one_graph_list[0] = adj_mat_add_self_loop(one_graph_list[0])
        #     # print(one_graph_list[0].size())
        #     # print(one_graph_list[0].device, one_graph_list[0].dtype, one_graph_list[1].device, one_graph_list[1].dtype)


    def built_model(self):
        self.model = self.model_(
            self.node_ft_size,
            self.label_num
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.epochs = 1000
        self.test_acc_list = []
        self.loss_op = torch.nn.BCEWithLogitsLoss()

    def train(self):
        self.model.train()
        total_loss = 0
        train_f1 = 0
        for _ in range(len(self.dataset.train_graphs)):
            one_train_graph = choice(self.dataset.train_graphs)
            idx = self.dataset.train_graphs.index(one_train_graph)

            self.optimizer.zero_grad()
            # 构建高阶邻居矩阵
            device_n_order_edge_list = []
            for adj_mat in self.train_n_order_adj_mat[idx]:
                adj_mat = adj_mat.to(self.device)

            pred = self.model(one_train_graph.node_ft , adj_mat).float()
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

    def metrics_model(self, graph_list, adj_mat_list):
        self.model.eval()
        total_micro_f1 = 0
        total_micro_p = 0
        total_micro_r = 0
        right_num = 0
        all_num = 0
        for idx in range(len(graph_list)):
            # cpu -> gpu
            # device_n_order_edge_list = []
            for adj_mat in adj_mat_list[idx]:
                adj_mat = adj_mat.to(self.device)

                # print(edge_list.size(), type(edge_list))

            with torch.no_grad():
                out = self.model(graph_list[idx].node_ft , adj_mat)
            pred = (out > 0).float().cpu()
            label = graph_list[idx].node_label.float().cpu()
            micro_f1 = metrics.f1_score(label, pred, average='micro')
            total_micro_p += metrics.precision_score(label, pred, average='micro')
            total_micro_r += metrics.recall_score(label, pred, average='micro')
            # print('node_num = ', one_graph.node_num, '1-pos = ', one_graph.node_label.sum(), 'micro_f1 = ', micro_f1)
            total_micro_f1 += metrics.f1_score(label, pred, average='micro')



        # return float(right_num) / all_num
        return total_micro_f1 / len(graph_list), total_micro_p / len(graph_list), total_micro_r / len(graph_list),

    def start(self, display=True):
        print('start train')
        for epoch in range(1, self.epochs+1):
            loss, train_f1 = self.train()

            # train_f1 = self.metrics_model(self.dataset.train_graphs)
            # valid_f1 = self.metrics_model(self.dataset.valid_graphs)
            test_f1, test_p, test_r = self.metrics_model(self.dataset.test_graphs, self.test_n_order_adj_mat)

            print('epoch = ', epoch, 'loss = ', loss, 'train_f1 = ', train_f1, 'test_f1 = ', test_f1, 'test_p = ', test_p, 'test_r = ', test_r)

if __name__ == '__main__':
    from GNN_Implement.model.modified_model.ppi_GCN2_V2_mat import GCN2
    from GNN_Implement.data_loader import PPI
    demo = NOrderInductiveLearningTestMat(GCN2, PPI())
    demo.start()