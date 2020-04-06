# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:25:16 2020

@author: DCMC
"""

import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
# from . import mask as maskUtils
import os
from collections import defaultdict
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

class HAHA:
    def __init__(self, annotation_file=None):
        # load dataset
        self.dataset = dict()
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            # assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
            
    def createIndex(self):
        # create index
        print('creating index...')
        caps = {}
        data_pair = []
        
        for cap in self.dataset:
            caps[cap['id']] = cap['caption']
            for sentence in cap['caption']:
                data_pair.append((cap['id'], sentence))
    
        print('index created!')
        
        # create class members
        self.caps = caps
        self.data_pair = data_pair
        
      
if __name__ == '__main__':
    from torch.autograd import Variable

    features = torch.rand(32, 16, 1799)
    captions = torch.LongTensor(32, 16).random_(1799)
    lengths = list(torch.LongTensor(32).random_(5, 17))
    lengths.sort(reverse=True)

    features, captions = Variable(features, requires_grad=True), Variable(captions)

    features = features+2

    print(features)
    print(captions)
    print(lengths)

    ll = CustomLoss()

    loss = ll(features, captions, lengths)
    loss.backward()

    print(loss)