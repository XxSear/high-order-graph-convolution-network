#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/4/7 21:23

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


from data_loader.classical_citation import ClassicalCitation

class Pubmed(ClassicalCitation):
    def __init__(self):
        super(Pubmed, self).__init__('Pubmed')

    @staticmethod
    def get_dataset():
        return Pubmed()
