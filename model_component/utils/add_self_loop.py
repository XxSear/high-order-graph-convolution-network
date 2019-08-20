#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/10 15:28

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
from model_component.utils.remove_mulit_edge import remove_mulit_edge

def list_add_self_loops(node_num, edges, remove=False):
    '''
    :param nodes: [num, feat_z]
    :param edges: [num, 2]
    :return:
    '''
    dtype, device = edges.dtype, edges.device
    index = torch.arange(node_num, dtype=dtype, device=device).view(-1, node_num)

    self_loop = torch.cat([index, index], 0)
    # print(index.size(), self_loop.size(), edges.size())
    edges = torch.cat([self_loop, edges], 1)
    if remove :
        return remove_mulit_edge(edges)
    return edges

def list_add_self_loops_(edges, remove=False, device=None):
    '''
    :param nodes: [num, feat_z]
    :param edges: [num, 2]
    :return:
    '''
    dtype = edges.dtype
    if device is None:
        device = edges.device
    node_num = torch.max(edges) + 1
    index = torch.arange(node_num, dtype=dtype, device=device).view(-1, node_num)

    self_loop = torch.cat([index, index], 0)
    # print(index.size(), self_loop.size(), edges.size())
    edges = torch.cat([self_loop, edges], 1)
    if remove :
        return remove_mulit_edge(edges)
    return edges

def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge_indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.unsqueeze(0).expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


def adj_mat_add_self_loop(adj_mat):
    device = adj_mat.device
    adj_mat = adj_mat + torch.eye(adj_mat.size()[0])
    return adj_mat

if __name__ == '__main__':
    demo = torch.tensor([
        [4, 0, 1],
        [1, 0, 0],
        [1, 3, 0]
    ]).float()

    print(adj_mat_add_self_loop(demo))