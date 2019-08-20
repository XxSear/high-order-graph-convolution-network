#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/2 11:51

base Info
"""
__author__ = '周朋飞'
__version__ = '1.0'


# import torch
#
# index = torch.LongStorage(4)
# index = torch.Tensor(1)
# print(index)

# demo = [1,2,3,4,5,6]
# print(demo[2:4])
# print(demo[4:7])

from torch_scatter import scatter_max
import torch

# src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
# index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
# out = src.new_zeros((2, 6))
#
# out, argmax = scatter_max(src, index, out=out)
#
# print(out)
# print(argmax)

# import torch
# from torch_geometric.utils import add_self_loops, softmax
# if __name__ == '__main__':
#     src = torch.tensor([[0.2, 0.4, 1.0, 1.5, 1.9, 6.0]],dtype=torch.float32).view(6,1)
#     index = torch.tensor([0,0,0,1,1,2]).view(6)
#     print(softmax(src, index, num_nodes=3))

# from data_loader import Pubmed


# dataset = Pubmed()
# train,valid,test= dataset.get_data_loader([4,4,2])
# for batch in train:
#     print(dataset)

# demo = [1,2,3,4]
# print(demo[-2:])