#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/15 18:47

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import matplotlib.pyplot as plt
from data_loader import Cora, Citeseer, Pubmed
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib



# degrees = torch.sum(adj_mat, dim=-1).numpy()


def display_first_degree(dataset, plot=True, pos=111):
    adj_mat = dataset.adj_table
    degrees = torch.sum(adj_mat, dim=-1).numpy()
    max_degree = int(np.max(degrees))
    if plot is True:
        plt.figure()
    else:
        plt.subplot(pos)
    plt.title(dataset.dataset_name + ' first order degree')
    plt.hist(degrees, bins=max_degree, log=True)
    # plt.plot()
    if plot is True:
        plt.show()
    else:
        plt.plot()


def display_second_degree(dataset, plot=True, pos=111):
    second_adj_mat = torch.mm(dataset.adj_table, dataset.adj_table)
    second_adj_mat[second_adj_mat > 0] = 1
    # print(second_adj_mat)
    degrees = torch.sum(second_adj_mat, dim=-1).numpy() - 1
    max_degree = int(np.max(degrees))
    if plot is True:
        plt.figure()
    else:
        plt.subplot(pos)
    plt.title(dataset.dataset_name + ' second order degree')
    plt.hist(degrees, bins=max_degree, log=True)
    if plot is True:
        plt.show()
    else:
        plt.plot()

def display_second_path(dataset, plot=True, pos=111):
    second_adj_mat = torch.mm(dataset.adj_table, dataset.adj_table)
    second_path = second_adj_mat[second_adj_mat > 0].numpy()
    max_path_num = int(np.max(second_path))
    # print(second_path, second_path[0])
    if plot is True:
        plt.figure()
    else:
        plt.subplot(pos)
    plt.title(dataset.dataset_name + ' second path num')
    plt.hist(second_path, bins=max_path_num, log=True)
    if plot is True:
        plt.show()
    else:
        plt.plot()



if __name__ == '__main__':
    dataset = Cora()
    # display_first_degree(dataset)
    # display_second_degree(dataset)
    display_second_path(dataset)