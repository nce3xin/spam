# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:32:22 2018

@author: nce3xin
"""

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append('..')
import hyperparams

import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(110240, hyperparams.cnn_out_dims)
        self.fc2 = nn.Linear(hyperparams.cnn_out_dims, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x=x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        
        # output intermediate layer 's result
        reserve=x
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1),reserve
