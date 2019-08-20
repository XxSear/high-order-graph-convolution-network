#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/7/25 11:09

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import collections
import os.path as osp
import os
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import torch.utils.data

processed_dir_name = 'processed'
raw_dir_name = 'raw'
dataset_obj_file_suffix = '.dat'


class BaseProcess(torch.utils.data.Dataset):
    def __init__(self, dataset_name, data_dir, reprocess):
        '''
        :param data_dir:
        :param reprocess:
        '''
        self.dataset_name = dataset_name
        self.data_dir = osp.expanduser(osp.normpath(data_dir))  # 数据集文件夹
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
            # print(all([osp.exists(name) for name in file_names]))
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
        ppi_obj_dat_path = osp.join(self.processed_dir, self.dataset_name+dataset_obj_file_suffix)
        return [ppi_obj_dat_path]

    def _load_processed_file(self):
        print('loading obj ', self.dataset_name)
        file_path = self._processed_file_list()[0]
        with open(file_path, 'rb') as f:
            # print(file_path)
            if sys.version_info > (3, 0):
                obj_dict = pickle.load(f)
            else:
                obj_dict = pickle.load(f)
        self.__dict__ = obj_dict

    def _save_processed_file(self):
        file_names = self._processed_file_list()
        for file_name in file_names:
            path = osp.join(self.dataset_dir, processed_dir_name, file_name)
            if dataset_obj_file_suffix in file_name:
                with open(path, 'wb') as f:
                    if sys.version_info > (3, 0):
                        pickle.dump(self.__dict__, f)
                    else:
                        pickle.dump(self.__dict__, f)
        print('save obj ', self.dataset_name)

