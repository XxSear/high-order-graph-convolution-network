#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/6/18 13:48

base Info
"""
__author__ = 'xx'
__version__ = '1.0'





import torch
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from data_loader import Cora, Pubmed, Citeseer
from model_component.utils.adj_mat import adj_2_bias_without_self_loop, adj_2_bias


class NOrderTest(object):
    def __init__(self, model_, dataset, order=3, aggr_depth=2, fliter_path_num=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        # self.dataset = Cora()
        # self.dataset = Citeseer()
        # self.dataset = Pubmed()
        self.model_ = model_
        self.dataset.to_device(self.device)
        self.epochs = 1000
        self.aggr_depth=aggr_depth
        self.order = order
        self.fliter_path_num = fliter_path_num

        self.n_order_adj_list = [self.dataset.add_self_loop_edge_list]
        for idx in range(1, self.order):
            self.n_order_adj_list.append(
                self.dataset.get_n_order_adj_list(idx, fliter_path_num=self.fliter_path_num, device=self.device))

        self.built_net()

    def built_net(self):
        self.model = self.model_(
            self.dataset.node_feature_size,
            self.dataset.label_size,
            aggr_depth=self.aggr_depth,
            order=self.order,
            mid_layer_channel=64
        ).to(self.device)

        self.optimizer_ = torch.optim.Adam


    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.dataset.all_node_feature,
                         self.n_order_adj_list)
        # out = self.model(self.dataset.all_node_feature,
        #                   self.dataset.bias_adj_mat)

        pred = out[self.dataset.train_index]

        loss = F.nll_loss(pred,
                          self.dataset.all_node_label[self.dataset.train_index])
        loss += self.l2_loss(0.005)
        # print(loss)
        loss.backward()
        self.optimizer.step()

    def l2_loss(self, l2_coef):
        loss_val = 0
        lambda_val = torch.tensor(1.)
        l2_reg = torch.tensor(0.)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        loss_val += lambda_val * l2_coef
        return loss_val

    def test(self):
        self.model.eval()
        pred = self.model(self.dataset.all_node_feature,
                          self.n_order_adj_list)
        pred = pred.max(1)[1]
        # print('pred.max(1) = ', pred)

        train_acc = pred[self.dataset.train_index].eq(
            self.dataset.all_node_label[self.dataset.train_index]).sum().float() / self.dataset.train_index.size()[0]

        val_acc = pred[self.dataset.valid_index].eq(
            self.dataset.all_node_label[self.dataset.valid_index]).sum().float() / self.dataset.valid_index.size()[0]

        test_acc = pred[self.dataset.test_index].eq(
            self.dataset.all_node_label[self.dataset.test_index]).sum().float() / self.dataset.test_index.size()[0]

        return train_acc, val_acc, test_acc

    def start(self, topn=10, display_acc=True):
        self.optimizer = self.optimizer_(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        # print(nodes.dtype)
        self.test_acc_list = []
        for epoch in range(1, self.epochs):
            self.train()
            # for i in self.model.parameters():
            #     print(i)
            train_acc, val_acc, test_acc = self.test()
            if display_acc:
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_acc, val_acc, test_acc))
            self.test_acc_list.append(test_acc.to('cpu').numpy().tolist())
        # 返回top 10 acc 的均值
        topn_acc = sorted(self.test_acc_list, reverse=True)[:topn]
        return sum(topn_acc) / len(topn_acc)


def ratio_test(model_, dataset, ratio, test_time=3):
    if dataset.dataset_name == 'Pubmed':
        aggr_depth = 2
    else:
        aggr_depth = 1

    model_res_list = []
    test_model = NOrderTest(model_, dataset, order=2, aggr_depth=aggr_depth, fliter_path_num=1)
    test_model.dataset.train_test_split(ratio, shuffle=False)

    for _ in range(test_time):
        test_model.built_net()
        res = test_model.start(display_acc=False)
        model_res_list.append(res)

    print(sum(model_res_list) / float(test_time))
    # print(ratio ,' ', sum(model_res_list) / float(test_time))

if __name__ == '__main__':
    from model.HGCN import HGCN

    torch.cuda.set_device(3)
    model_ = HGCN

    ratio_list = [
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

    for dataset in [Cora(), Citeseer(), Pubmed()]:
        print(dataset.dataset_name)
        for ratio in ratio_list:
            ratio_test(model_, dataset, ratio)