#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/5/16 13:53

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

from data_loader import Cora,Citeseer,Pubmed
import networkx as nx
import numpy as np
import torch

class Graph_stat():
    def __init__(self, dataset):
        self.dataset= dataset
        # self.dataset = Cora()
        self.edges = self.dataset.edge_index.numpy()
        self.graph = nx.Graph()
        for i, j in zip(self.edges[0], self.edges[1]):
            self.graph.add_edge(i, j)

        print( self.dataset.dataset_name)

        self.degree =  nx.degree_histogram(self.graph)

        self.average_clustering = nx.average_clustering(self.graph)
        print('average_clustering = ',self.average_clustering)

        # self.average_neighbor_degree = nx.average_neighbor_degree(self.graph)
        # print('average_neighbor_degree = ', self.average_neighbor_degree)

        self.average_degree = self.dataset.edge_index.size()[1] / self.dataset.node_size
        print('average_degree = ', self.average_degree)

        self.average_second_degree = self.get_average_second_degree()
        print('average_second_degree = ', self.average_second_degree)


        # print(nx.numeric_assortativity_coefficient(self.graph,'size'))

    def get_average_second_degree(self):
        second_adj_mat = torch.mm(self.dataset.adj_table, self.dataset.adj_table)
        second_adj_mat[second_adj_mat > 0] = 1
        degrees = torch.sum(second_adj_mat, dim=-1).numpy() - 1
        average_degree = np.mean(degrees)
        return average_degree


class NGraph_stat():
    def __init__(self, dataset):
        self.dataset= dataset
        # self.dataset = Cora()
        self.graphs = self.dataset.all_graph
        self.average_clustering_list = []
        self.average_degree_list = []
        self.shortest_path = []

        for graph in self.graphs:
            self.stat_one_graph(graph)

        print( float(sum(self.average_clustering_list)) / len(self.average_clustering_list) )
        print( float(sum(self.average_degree_list)) / len(self.average_degree_list) )
        # print(sum(self.shortest_path) / len(self.shortest_path))

    def stat_one_graph(self, graph):
        graph_edges = graph.edge_list.numpy()
        node_size = graph.node_ft.size()[0]
        # print(graph_edges)
        tmp_graph = nx.Graph()
        for i, j in zip(graph_edges[0], graph_edges[1]):
            tmp_graph.add_edge(i, j)

        average_clustering = nx.average_clustering(tmp_graph)
        # print(average_clustering)
        self.average_clustering_list.append(average_clustering)
        average_degree =  tmp_graph.number_of_edges() * 2.0 / tmp_graph.number_of_nodes()
        self.average_degree_list.append(average_degree)

        # print(average_clustering, average_degree)

        # shortest_path = nx.average_shortest_path_length(tmp_graph)
        # self.shortest_path.append(shortest_path)

if __name__ == '__main__':
    # Graph_stat(Cora())
    # Graph_stat(Citeseer())
    # Graph_stat(Pubmed())
    from GNN_Implement.data_loader import PPI
    NGraph_stat(PPI())
