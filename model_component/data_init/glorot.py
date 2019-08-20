#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/8 12:12

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
import torch
import math

'''
https://blog.csdn.net/u012151283/article/details/78230891
'''


def glorot_normal(tensor):
    std = math.sqrt(2.0 / (tensor.size(-2) + tensor.size(-1)))
    mean = 0
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    return tensor

def glorot_uniform(tensor):
    bound = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    # print(bound)
    torch.nn.init.uniform_(tensor, a=-bound, b=bound)
    return tensor

if __name__ == '__main__':
    tensor = torch.empty(3, 4)
    print(glorot_normal(tensor))
    print(glorot_uniform(tensor))