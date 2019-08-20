#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/7 21:23

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


from data_loader.classical_citation import ClassicalCitation

class Citeseer(ClassicalCitation):
    def __init__(self):
        super(Citeseer, self).__init__('Citeseer')

    @staticmethod
    def get_dataset():
        return Citeseer()

if __name__ == '__main__':
    Citeseer()
