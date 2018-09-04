# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:53:23 2018

@author: nce3xin
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from loaddata.load_dataset import load_data

num_segment=3

train_pt='data/train.arff'
test_pt='data/test.arff'
train_loader,test_loader,train_dataset,test_dataset=load_data(train_pt,test_pt)
dataset_size={'train':len(train_dataset),'test':len(test_dataset)}

num_features=train_dataset.get_in_ftrs()
num_classes=train_dataset.get_n_class()

if num_features < num_segment:
    num_segment=num_features
    print('num_features < num_segment, therefore we modify num_segment to num_features.')

# network definition
class Model(nn.Module):
    def __init__(self,num_segment=num_segment):
        super(Model,self).__init__()
        self.num_segment=num_segment
        self.branches=[]
        # several ModelA
        for i in range(num_segment):
            if i==num_segment-1 and num_features%num_segment!=0:
                n_in=num_features % num_segment + int(num_features / num_segment)
            else:
                n_in=int(num_features/num_segment)
            self.branches.append(ModelA(n_in,2).float())
        # ModelB
        input_size=torch.tensor([c.input_size for c in self.branches]).sum().item()
        self.upper_layer=ModelB(input_size,num_classes,1)
        
    def forward(self,x):
        output=[]
        datasets_list=self.make_datasets_list(x)
        
        # several ModelA
        for i,branch in enumerate(self.branches):
            output.append(branch(datasets_list[i]))
        # ModelB
        x=torch.cat(output,1)
        x=x.view(x.size()[0],-1)
        x=self.upper_layer(x)
        return x
    
    def make_datasets_list(self,x):
        datasets_list=[]
        last_idx=0
        for i in range(num_segment):
            if i==num_segment-1 and num_features%num_segment!=0:
                l=num_features%num_segment
                out=x.view(x.size()[0],-1)[:,last_idx:]
            else:
                l=int(num_features/num_segment)
                out=x.view(x.size()[0],-1)[:,i*l:(i+1)*l]
                last_idx=(i+1)*l
            datasets_list.append(out)
        return datasets_list
    

class ModelB(nn.Module):
    def __init__(self,input_size,num_classes,num_layers):
        super(ModelB,self).__init__()
        self.num_layers=num_layers
        self.linear1=nn.Linear(input_size,input_size)
        self.linear2=nn.Linear(input_size,num_classes)
        
    def forward(self,x):
        for _ in range(self.num_layers):
            x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x

class ModelA(nn.Module):
    def __init__(self,input_size,num_layers):
        super(ModelA,self).__init__()
        self.num_layers=num_layers
        self.input_size=input_size
        self.linear=nn.Linear(input_size,input_size)
    
    def forward(self,x):
        for _ in range(self.num_layers):
            x=F.relu(self.linear(x))
        return x