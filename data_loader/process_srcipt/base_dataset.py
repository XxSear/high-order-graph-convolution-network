#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/2 14:24

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import collections
import os.path as osp
import os

import torch.utils.data
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle



processed_dir_name = 'processed'
raw_dir_name = 'raw'
dataset_obj_file_suffix = '.dat'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, reprocess):
        '''
        :param data_dir:
        :param reprocess:
        '''
        self.data_dir = osp.expanduser(osp.normpath(data_dir))
        self.raw_dir = osp.join(self.data_dir, raw_dir_name)
        self.processed_dir = osp.join(self.data_dir, processed_dir_name)
        self.processed_data = None
        self.reprocess = reprocess
        self._get_data()

    def _get_data(self):
        # print('_get_data')
        # print(osp.exists(self.processed_dir))
        # print('self.processed_dir = ', self.processed_dir)
        if osp.exists(self.processed_dir):
            file_names = self._processed_file_list()
            if all([osp.exists(name) for name in file_names]) and self.reprocess is False:
                self._load_processed_file()
            else:
                self._process()
        else:
            # os.makedirs(self.data_dir)
            os.mkdir(self.processed_dir)
            self._process()

    def _process(self):
        raise NotImplementedError

    def _processed_file_list(self):
        raise NotImplementedError

    def load_processed_file(self):
        raise NotImplementedError


