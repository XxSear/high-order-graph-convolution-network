#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/12 18:31

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch


def index_2_mask(num, index):
    mask = torch.zeros(num).long()
    mask[index] = 1
    # print(mask)
    return mask

def mask_2_index(mask):
    return mask.nonzero().squeeze()

if __name__ == '__main__':
    demo = torch.tensor([2,1,5,16], dtype=torch.long)
    res = index_2_mask(17, demo)
