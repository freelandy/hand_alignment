# coding:utf-8

import pickle

class Model:
    R = []
    m = []
    file_name = ''

    def __init__(self,file_name):
        self.file_name = file_name
        self.R = []
        self.m = []

    def save(self):
        with open(self.file_name,'wb') as fp:
            pickle.dump(self,fp)

        print('File saved.')

    def load(self):
        with open(self.file_name,'rb') as fp:
            model = pickle.load(fp)

        return model