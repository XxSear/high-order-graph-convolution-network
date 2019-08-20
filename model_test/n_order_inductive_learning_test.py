#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/29 11:10

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
from GNN_Implement.model_component.utils.add_self_loop import list_add_self_loops_

class NOrderInductiveLearningTest(object):
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
        self.get_n_order_edge_list()
        self.built_model()


    # 图太多 放到CPU中
    def get_n_order_edge_list(self, device='cpu'):
        print('bulit n order edge list')
        self.train_n_order_edge_list = [[graph.edge_list] for graph in self.dataset.train_graphs]
        self.valid_n_order_edge_list = [[graph.edge_list] for graph in self.dataset.valid_graphs]
        self.test_n_order_edge_list = [[graph.edge_list] for graph in self.dataset.test_graphs]

        for one_graph_list in chain(self.train_n_order_edge_list,
                                    self.valid_n_order_edge_list,
                                    self.test_n_order_edge_list):
            for order_idx in range(1, self.adj_order):
                high_order_adj_list = adj_list_to_n_order_adj_list(one_graph_list[0], order_idx, device=device, fliter_path_num=9)
                one_graph_list.append(high_order_adj_list)
                print('one_graph_list[0] size = ', one_graph_list[0].size(), 'high_order_adj_list size = ', high_order_adj_list.size())

            # 加入 self--loop
            # print(one_graph_list[0].size())
            one_graph_list[0] = list_add_self_loops_(one_graph_list[0], device='cpu')
            # print(one_graph_list[0].size())
            # print(one_graph_list[0].device, one_graph_list[0].dtype, one_graph_list[1].device, one_graph_list[1].dtype)



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
            for edge_list in self.train_n_order_edge_list[idx]:
                device_n_order_edge_list.append(edge_list.to(self.device))
                # print(edge_list.size(), type(edge_list), edge_list.device)

            pred = self.model(one_train_graph.node_ft , device_n_order_edge_list).float()
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

    def metrics_model(self, graph_list, n_order_edge_lists):
        self.model.eval()
        total_micro_f1 = 0
        total_micro_p = 0
        total_micro_r = 0
        right_num = 0
        all_num = 0
        for idx in range(len(graph_list)):
            # cpu -> gpu
            device_n_order_edge_list = []
            for edge_list in n_order_edge_lists[idx]:
                device_n_order_edge_list.append(edge_list.to(self.device))
                # print(edge_list.size(), type(edge_list))

            with torch.no_grad():
                out = self.model(graph_list[idx].node_ft , device_n_order_edge_list)
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
            test_f1, test_p, test_r = self.metrics_model(self.dataset.test_graphs, self.test_n_order_edge_list)

            print('epoch = ', epoch, 'loss = ', loss, 'train_f1 = ', train_f1, 'test_f1 = ', test_f1, 'test_p = ', test_p, 'test_r = ', test_r)

            # if display is True:
            #     print('Epoch: {:02d}, Loss: {:.4f}, train_f1: {:.4f}, train_f1: {:.4f}, train_f1: {:.4f}'.format(
            #         epoch, loss, valid_f1, valid_f1, test_f1))

if __name__ == '__main__':
    from GNN_Implement.model.modified_model.ppi_GCN2 import GCN2
    from GNN_Implement.model.modified_model.ppi_GAT2_V1 import GAT2

    # demo_test = InductiveLearningTest(GAT, PPI())
    # demo_test = NOrderInductiveLearningTest(GCN2, PPI())
    # demo_test = InductiveLearningTest(GCN2, PPI())

    from GNN_Implement.model.modified_model.ppi_GCN2_V2 import GCN2
    torch.cuda.set_device(2)
    demo_test = NOrderInductiveLearningTest(GCN2, PPI())
    # demo_test = NOrderInductiveLearningTest(GAT2, PPI())
    # print(demo_test.dataset.train_graphs[0].node_label.device)

    demo_test.start()


