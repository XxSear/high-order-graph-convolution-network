#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/8 12:10

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
'''
Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。
Xavier初始化的推导过程是基于线性函数的，但是它在一些非线性神经元中也很有效。
https://blog.csdn.net/lanchunhui/article/details/70318941

torch.nn.init.xavier_uniform_(tensor, gain=1)
torch.nn.init.xavier_normal_(tensor, gain=1)
'''