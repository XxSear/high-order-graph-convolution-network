#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/15 13:44

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch


def mat_2_edge_list(torch_mat, device):
    mat =  torch_mat.nonzero().squeeze()
    long_mat = torch.transpose(mat, 1, 0).to(device)
    # print(long_mat.dtype)
    return long_mat


def mulit_mat_2_edge_list(torch_adj_mats, device):
    edge_list = []
    for index in range(torch_adj_mats.size()[0]):
        edge_list.append( mat_2_edge_list(torch_adj_mats[index], device))
    return edge_list


def edge_list_2_adj_mat(edge_list, node_num=None, device='cpu', direction=False):
    if node_num is None:
        node_num = torch.max(edge_list).long() + 1

    degree_mat = torch.zeros(node_num, node_num).to(device)
    for idx in range(edge_list.size()[-1]):
        x = edge_list[0][idx]
        y = edge_list[1][idx]
        degree_mat[x, y] = 1
        if direction is True:
            degree_mat[y, x] = 1

    return degree_mat





"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
# 给邻接矩阵做变换 有边--0  无边---极小值
def mulit_adj_2_bias(adj_mats):
    bias_mats = torch.zeros(adj_mats.size())
    # print(bias_mats.size())
    for graph_index in range(adj_mats.size()[0]):
        bias_mats[graph_index,:,:] = adj_2_bias(adj_mats[graph_index])
    return bias_mats


def adj_2_bias(adj):
    node_size = adj.size()[0]
    new_mat = ((torch.eye(node_size).to(adj.device) + adj) >= 1).float()
    # print(new_mat)
    new_mat = torch.tensor(-1e9) * (1 - new_mat)
    # print(new_mat)
    return new_mat


def adj_2_bias_without_self_loop(adj):
    node_size = adj.size()[0]
    new_mat = (adj >= 1).float()
    # print(new_mat)
    new_mat = torch.tensor(-1e9).to(adj.device) * (1 - new_mat)
    # print(new_mat)
    return new_mat


def filter_adj_mat(adj, num):
    adj[adj <= num] = 0
    # print(adj)
    return adj


def get_degree_mat(adj_mat, pow=1, save_gpu_memory=True):
    if save_gpu_memory:
        original_device = adj_mat.device
        cal_device = 2
        adj_mat = adj_mat.to(cal_device)
    else:
        cal_device = adj_mat.device
    degree_mat = torch.eye(adj_mat.size()[0]).cuda(cal_device)
    degree_list = torch.sum((adj_mat > 0), dim = 1).float()
    degree_list = torch.pow(degree_list, pow)

    degree_mat = degree_mat * degree_list
    # degree_mat = torch.pow(degree_mat, pow)
    # degree_mat[degree_mat == float("Inf")] = 0

    if save_gpu_memory:
        degree_mat = degree_mat.to(original_device)
        # print(degree_mat.device)
    return degree_mat


def get_laplace_mat(adj_mat, type='sym'):
    if type == 'sym':
        # Symmetric normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5)

        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)
        return laplace_mat
    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)

        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        return laplace_mat


def adj_mat_to_n_order_adj_list(adj_mat, order_num, fliter_path_num='mean', device='cpu'):
    n_order_adj_mat = adj_mat
    for i in range(1, order_num):
        n_order_adj_mat = torch.mm(n_order_adj_mat, adj_mat)

    if fliter_path_num == 'mean':
        # print(torch.mean(n_order_adj_mat[n_order_adj_mat > 0]))
        fliter_path_num = torch.mean(n_order_adj_mat[n_order_adj_mat > 0]) * 5
        # print(n_order_adj_mat[n_order_adj_mat > 0])

    # print(n_order_adj_mat.nonzero().size())
    n_order_adj_list = (n_order_adj_mat > fliter_path_num).nonzero()  # 不加入self-loop 过滤邻居
    n_order_adj_list = torch.transpose(n_order_adj_list, 0, 1)
    n_order_adj_list = n_order_adj_list.to(device)
    return n_order_adj_list


def adj_list_to_n_order_adj_list(adj_list, order_num, fliter_path_num='mean', device='cpu'):
    adj_mat = edge_list_2_adj_mat(adj_list, device=device)
    # print('adj_mat size = ', adj_mat.size())
    n_order_adj_list = adj_mat_to_n_order_adj_list(adj_mat, order_num=order_num,
                                                   fliter_path_num=fliter_path_num, device=device)
    return n_order_adj_list

def adj_mat_to_n_order_adj_mat(adj_mat, order_num, fliter_path_num='mean', device='cpu'):
    n_order_adj_mat = adj_mat
    for idx in range(1, order_num):
        n_order_adj_mat = torch.mm(n_order_adj_mat, adj_mat)

    if fliter_path_num == 'mean':
        # print(torch.mean(n_order_adj_mat[n_order_adj_mat > 0]))
        fliter_path_num = torch.mean(n_order_adj_mat[n_order_adj_mat > 0])

    n_order_adj_mat[n_order_adj_mat <= fliter_path_num] = 0
    return n_order_adj_mat



if __name__ == '__main__':
    demo = torch.tensor([
        [4, 0, 1],
        [1, 0, 0],
        [1, 3, 0]
    ]).float()

    # edge_list = mat_2_edge_list(demo, device='cpu')
    # print(edge_list, edge_list.size())
    #
    # adj_list = adj_list_to_n_order_adj_list(edge_list, 2)
    # print(adj_list)

    print(adj_mat_to_n_order_adj_mat(demo, 2, fliter_path_num=1))


    # mat2 = 1 - demo

    # print(adj_2_bias(demo))

    # res = mat_2_edge_list(demo)
    # print(res)

    # adj_mats = torch.cat([demo.view(-1, 3, 3), mat2.view(-1, 3, 3)], 0)
    # # print(adj_mats.size())
    # res = mulit_adj_2_bias(adj_mats)
    # # for mat in res:
    # #     print(mat)
    # print(res)

    # print(get_degree_mat(demo, -0.5))
    # print(get_laplace_mat(demo))
    # print(torch.cuda.get_device_capability(0))