#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/10 16:02

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import numpy as np

def remove_mulit_edge(edges):
    dtype, device = edges.dtype, edges.device
    # print(device)
    if device != 'cpu':
        edges = edges.cpu()
    np_edges = edges.numpy()
    new_np_edges = np.unique(np_edges, axis=0)
    edges = torch.from_numpy(new_np_edges).to(device)
    return edges


if __name__ == '__main__':
    data = torch.tensor([[0, 0],
        [1, 1],
        [1, 1],
        [2, 2],
        [0, 1],
        [0, 2],
        [1, 2]], dtype=torch.int32).to('cuda')

    print(remove_mulit_edge(data))