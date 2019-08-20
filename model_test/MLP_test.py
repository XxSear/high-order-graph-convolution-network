#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/8/1 1:01

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import torch.nn.functional as F
from GNN_Implement.model.modified_model.MLP import MLP

class MLP_test(object):
    def __init__(self, dataset, layer=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(3)
        self.dataset = dataset
        self.layer = layer
        self.dataset.to_device(self.device)
        self.built_net()

    def built_net(self):
        self.model = MLP(
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
            layer=self.layer
        ).to(self.device)
        self.optimizer_ = torch.optim.Adam
        self.epochs = 1000
        self.test_acc_list = []
        if self.dataset.dataset_name == 'ppi':
            self.loss_op = torch.nn.BCEWithLogitsLoss
        else:
            self.loss_op =  F.nll_loss

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.dataset.all_node_feature)
        # print(type(self.dataset.train_index))
        # print(type(out), type(self.dataset.all_node_label))

        loss = self.loss_op(out[self.dataset.train_index],
                   self.dataset.all_node_label[self.dataset.train_index])
        loss.backward()
        self.optimizer.step()

    def test(self):
        self.model.eval()
        # print(self.model.training)
        pred = self.model(self.dataset.all_node_feature)
        # print('pred.max(1) = ',pred.max(1))
        pred = pred.max(1)[1]
        # print(pred)

        train_acc = pred[self.dataset.train_index].eq(
            self.dataset.all_node_label[self.dataset.train_index]).sum().float() / self.dataset.train_index.size()[0]

        val_acc = pred[self.dataset.valid_index].eq(
            self.dataset.all_node_label[self.dataset.valid_index]).sum().float() / self.dataset.valid_index.size()[0]

        test_acc = pred[self.dataset.test_index].eq(
            self.dataset.all_node_label[self.dataset.test_index]).sum().float() / self.dataset.test_index.size()[0]

        # if test_acc > self.best_acc:
        #     self.best_acc = test_acc
        return train_acc, val_acc, test_acc

    def start(self, topn=10, display_acc=True):
        # print(nodes.dtype)
        self.optimizer = self.optimizer_(self.model.parameters(), lr=0.005, weight_decay=5e-4)
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
        if display_acc:
            print('top ', topn, 'mean acc = ', sum(topn_acc) / len(topn_acc))
        return sum(topn_acc) / len(topn_acc)


if __name__ == '__main__':
    from GNN_Implement.data_loader import Cora, Pubmed, Citeseer, PPI

    torch.cuda.set_device(2)
    # dataset = Cora()
    # dataset = Citeseer()

    dataset = Pubmed()
    # dataset = PPI()
    test_model = MLP_test(dataset)
    test_model.dataset.train_test_split([0.8, 0.1, 0.1], shuffle=False)
    test_model.start()