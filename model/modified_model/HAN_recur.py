#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/13 14:50

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from data_loader import ACM_3025
from GNN_Implement.model_component.conv.gat_conv_input_list import GatConv
from GNN_Implement.model_component.conv.aggregation_attn_conv import AggregatioAttnConv


class HAN(torch.nn.Module):
    def __init__(self, node_ftr_size, n_classes,
                 attn_drop=0.6,
                 ffd_drop=0.6,
                 node_level_attn_head=8,
                 node_level_attn_layer=1,
                 node_level_attn_out_size=8,
                 semantic_level_head=1,
                 semantic_attn_size=128,
                 device='cpu'):
        super(HAN, self).__init__()
        self.node_ftr_size = node_ftr_size
        self.n_classes = n_classes
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.node_level_attn_head = node_level_attn_head
        self.node_level_attn_layer = node_level_attn_layer
        self.node_level_attn_out_size = node_level_attn_out_size
        self.semantic_level_head = semantic_level_head
        self.semantic_attn_size = semantic_attn_size
        self.device = device

        self.node_level_convs = []
        for i in range(self.node_level_attn_layer):
            if i == 0:
                node_attn_conv = GatConv(self.node_ftr_size,
                                         self.node_level_attn_out_size,
                                         heads=self.node_level_attn_head,
                                         dropout=self.attn_drop)
            else:
                node_attn_conv = GatConv(self.node_level_attn_out_size * self.node_level_attn_head,
                                         self.node_level_attn_out_size,
                                         heads=self.node_level_attn_head,
                                         dropout=self.attn_drop)
                # node_attn_conv.to(device)
            self.node_level_convs.append(node_attn_conv)

        self.semantic_attn = AggregatioAttnConv(self.node_level_attn_out_size * self.node_level_attn_head,
                                           self.semantic_attn_size)

        self.linear = torch.nn.Linear(
            self.semantic_attn_size * self.semantic_level_head, self.n_classes)

    def forward(self, node_feature, edges_list):
        # print(next(self.parameters()).is_cuda)
        # print('node_feature = ',node_feature[10])
        # print(node_feature[30])
        node_feature = node_feature.float()
        node_size = node_feature.size()[0]
        embed_list = []
        for graph_index in range(len(edges_list)):
            x = node_feature
            for attn_layer in self.node_level_convs:
                # x = F.dropout(x, training=self.training)
                attn_layer.to(self.device)
                x = attn_layer(x, edges_list[graph_index])
            embed_list.append(x.view(1, node_size, -1))

        embed_list = torch.cat(embed_list, 0)
        # print(embed_list)
        # embed_list  [n_graph, n_node, self.attn_out_size * self.node_level_attn_head]
        final_embed, attval = self.semantic_attn(embed_list)
        # return final_embed

        out = self.linear(final_embed)
        # print(out)
        # out = torch.div(out, self.semantic_level_head)
        return F.log_softmax(out, dim=-1)


def train(model, data):
    model.train()
    # print(data.adj_mats[1])
    out = model(data.node_feature, data.edges_list)
    pred = out[data.train_index]
    # print('pred = ', pred.size(), pred)
    # print('label = ', data.train_labels.size())
    optimizer.zero_grad()
    loss = F.nll_loss(pred, data.train_labels)
    # print(loss)
    loss.backward()
    optimizer.step()

def test(model, data):
    model.eval()
    pred = model(data.node_feature, data.edges_list).max(1)[1]
    # print(pred)
    train_acc = pred[data.train_index].eq(data.train_labels).sum().float() / data.train_labels.size()[0]
    val_acc = pred[data.val_index].eq(data.valid_labels).sum().float() / data.valid_labels.size()[0]
    test_acc = pred[data.test_index].eq(data.test_labels).sum().float() / data.test_labels.size()[0]
    return [train_acc, val_acc, test_acc]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    data = ACM_3025(device=device)
    # data.node_feature, data.adj_mats
    # print(data.adj_mats.size())
    # print(data.edges_list[0].size(), data.edges_list[1].size())
    print(data.edges_list[0].dtype)
    print(data.edges_list[1].size())

    model = HAN(
        node_ftr_size=data.node_feature_size,
        n_classes=data.label_size,
        device=device
        )
    model = model.to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    # print(model.parameters())
    # train(model, data)
    for epoch in range(1, 200):
        train(model, data)
        print('Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, *test(model, data)))
