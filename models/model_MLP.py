# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:50:40 2018

@author: nce3xin
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,input_size,num_classes):
        super(Net,self).__init__()
        self.linear1=nn.Linear(input_size,128)
        self.linear2=nn.Linear(128,256)
        self.linear3=nn.Linear(256,512)
        self.linear4=nn.Linear(512,256)
        self.linear5=nn.Linear(256,128)
        self.linear6=nn.Linear(128,num_classes)
        
    def forward(self,x):
        x=x.float()
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x=F.relu(self.linear3(x))
        x=F.relu(self.linear4(x))
        x=F.relu(self.linear5(x))
        x=self.linear6(x)
        return F.log_softmax(x,dim=1)