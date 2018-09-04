# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:28:56 2018

@author: nce3xin
"""
from __future__ import print_function

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append('..')


import torch
from scipy.io import arff
import pandas as pd
import hyperparams
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import extract_2D_features
#from CNN_res import extract_dense_representation
import numpy as np
from models import model_CNN
import hyperparams

batch_size = hyperparams.batch_size

use_cuda = not hyperparams.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# CNN_res
def load_model(pt):
    model=model_CNN.CNNModel()
    model.load_state_dict(torch.load(pt))
    return model

# load CNN model
CNN_model_pt='models_saved/CNN_cuda_epoch=90_outdims=25.pth'
#CNN_model_pt='models_saved/CNN_cpu_epoch=6_outdims=5.pth'

if hyperparams.MODEL!='CNN' and hyperparams.CNN_mapping:
    model=load_model(CNN_model_pt)
    model=model.to(device)

# load merged_train_df and merged_test_df
merged_train_df=pd.read_csv('data/gen/train.csv')
merged_test_df=pd.read_csv('data/gen/test.csv')

def extract_dense_ftrs_by_CNN(model,data_loader,device):
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            data = data.float()
            output,dense=model(data)
            if i==0:
                prev=dense
            else:
                prev=torch.cat([prev,dense],0)
    return prev

def extract_dense_representation():
    train_loader,test_loader,_,_=load_2d_data_for_CNN(shuffle=False)
    
    train_dense=extract_dense_ftrs_by_CNN(model,train_loader,device)
    test_dense=extract_dense_ftrs_by_CNN(model,test_loader,device)
    return train_dense,test_dense
# CNN_res





class WeiboDenseDataset(torch.utils.data.Dataset):
    def __init__(self,data,normalization):
        self.normalization=normalization
        
        self.ftrs=data[:,1:-1] # remove "Instance number" column
        #self.ftrs=self.ftrs.float()
        
        if self.normalization:
            if hyperparams.standard_scale:
                self.ftrs=preprocessing.scale(self.ftrs)
            elif hyperparams.min_max_scaler:
                min_max_scaler=MinMaxScaler()
                self.ftrs=min_max_scaler.fit_transform(self.ftrs)
        
        self.label=data[:,-1].astype(int)
        
    def __len__(self):
        return len(self.ftrs)
    
    def __getitem__(self,idx):
        ftrs=self.ftrs[idx,:]
        label=self.label[idx] 
        return (ftrs,label)   
    
    def get_n_class(self):
        return 3
    
    def get_in_ftrs(self):
        return self.ftrs.shape[1]
    

class Weibo2DForCNN(torch.utils.data.Dataset):
    def __init__(self,ftrs2D,labels_for_2D_ftrs):
        self.ftrs=ftrs2D
        self.ftrs=self.ftrs.unsqueeze(1)
        self.ftrs=self.ftrs.float()
        self.label=labels_for_2D_ftrs
        
    def __len__(self):
        return len(self.ftrs)
    
    def __getitem__(self,idx):
        ftrs=self.ftrs[idx,:]
        label=self.label[:,idx] # Because label is a horizontal vector
        return (ftrs,label)   
    
    def get_n_class(self):
        return 3
    
    def get_in_ftrs(self):
        return self.ftrs.size()[1]

class WeiboTemporalDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,normalization):
        self.df=pd.read_csv(file_path)
        self.normalization=normalization
        
        # convert object labels to numeric
        self.df['Class']=self.df['Class'].astype('category')
        self.df['Class']=self.df['Class'].cat.rename_categories([0,1,2]).astype(int)
        
        self.df=self.df.iloc[:,1:]
        self.ftrs=self.df.iloc[:,:-1]  # features
        
        self.ftrs=self.ftrs.values
        
        if self.normalization:
            if hyperparams.standard_scale:
                self.ftrs=preprocessing.scale(self.ftrs)
            elif hyperparams.min_max_scaler:
                min_max_scaler=MinMaxScaler()
                self.ftrs=min_max_scaler.fit_transform(self.ftrs)
        
        self.ftrs=torch.from_numpy(self.ftrs)
        self.ftrs=self.ftrs.float()
        self.label=torch.LongTensor([self.df.iloc[:,-1]])  # labels

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self,idx):
        ftrs=self.ftrs[idx,:]
        label=self.label[:,idx] # Because label is a horizontal vector
        return (ftrs,label)    
    
    def get_n_class(self):
        return 3
    
    def get_in_ftrs(self):
        return self.ftrs.size()[1]
    
# dataset
class WeiboDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,temporal,normalization):
        # read .arff file
        data = arff.loadarff(file_path)
        self.df = pd.DataFrame(data[0])
        # convert object labels to numeric
        self.df['Class']=self.df['Class'].astype('category')
        self.df['Class']=self.df['Class'].cat.rename_categories([0,1,2]).astype(int)
        self.df=self.df.iloc[:,1:] # remove index column, reserve label column (the last column)
        
        self.temporal=temporal
        self.normalization=normalization
        
        if self.temporal:
            self.ftrs=self.df.iloc[:,:-1]  # features
        else:
            columns=[0,1,2,3,4,5]
            self.ftrs=self.df.iloc[:,columns]  # features
        
        if self.normalization:
            columns=self.ftrs.columns # select columns to normalize
            scaler=MinMaxScaler()
            self.ftrs[columns]=scaler.fit_transform(self.ftrs[columns])
        
        self.ftrs=torch.from_numpy(self.ftrs.values)
        self.ftrs=self.ftrs.float()
        self.label=torch.LongTensor([self.df.iloc[:,-1]])  # labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        ftrs=self.ftrs[idx,:]
        label=self.label[:,idx] # Because label is a horizontal vector
        return (ftrs,label)
    
    def get_n_class(self):
        return 3
    
    def get_in_ftrs(self):
        return self.ftrs.size()[1]

def load_data(train_pt,test_pt):
    temporal=True
    normalization=False
    train_dataset=WeiboDataset(train_pt,temporal,normalization)
    test_dataset=WeiboDataset(test_pt,temporal,normalization)
    train_loader=torch.utils.data.DataLoader(dataset = train_dataset,
                            batch_size=batch_size,
                            shuffle = True)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=True)
    return train_loader,test_loader,train_dataset,test_dataset

def load_temporal_data(train_pt,test_pt,normalization):
    train_dataset=WeiboTemporalDataset(train_pt,normalization)
    test_dataset=WeiboTemporalDataset(test_pt,normalization)
    train_loader=torch.utils.data.DataLoader(dataset = train_dataset,
                            batch_size=batch_size,
                            shuffle = True)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=True)
    return train_loader,test_loader,train_dataset,test_dataset

def load_2d_data_for_CNN(shuffle=True):
    train_2d_features,test_2d_features=extract_2D_features.extract_2D_features()
    train_labels_for_2D_ftrs,test_labels_for_2D_ftrs=extract_2D_features.extract_labels_for_2D_features()
    
    train_dataset=Weibo2DForCNN(train_2d_features,train_labels_for_2D_ftrs)
    test_dataset=Weibo2DForCNN(test_2d_features,test_labels_for_2D_ftrs)
    
    train_loader=torch.utils.data.DataLoader(dataset = train_dataset,
                            batch_size=batch_size,
                            shuffle = shuffle)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=shuffle)
    
    return train_loader,test_loader,train_dataset,test_dataset


def load_cnn_dense():
    # load merged_train_df and merged_test_df
    merged_train_df=pd.read_csv('data/gen/train.csv')
    merged_test_df=pd.read_csv('data/gen/test.csv')
    
    train_label=merged_train_df.iloc[:,-1]
    test_label=merged_test_df.iloc[:,-1]
    
    # convert object labels to numeric
    train_label=train_label.astype('category')
    train_label=train_label.cat.rename_categories([0,1,2]).astype(int)
    test_label=test_label.astype('category')
    test_label=test_label.cat.rename_categories([0,1,2]).astype(int)
    
    merged_train_df=merged_train_df.iloc[:,:13]
    merged_test_df=merged_test_df.iloc[:,:13]
    
    merged_train_df=merged_train_df.values
    merged_test_df=merged_test_df.values
    
    train_label=train_label.values
    test_label=test_label.values
    train_label=train_label.reshape(train_label.shape[0],1) # reshape
    test_label=test_label.reshape(test_label.shape[0],1) # reshape
    
    train_dense,test_dense=extract_dense_representation()
    
    train_dense=train_dense.cpu().numpy()
    test_dense=test_dense.cpu().numpy()
    
    train_data=np.concatenate((merged_train_df,train_dense),axis=1)
    train_data=np.concatenate((train_data,train_label),axis=1)
    test_data=np.concatenate((merged_test_df,test_dense),axis=1)
    test_data=np.concatenate((test_data,test_label),axis=1)
    
    normalization=hyperparams.normalization
    
    train_dataset=WeiboDenseDataset(train_data,normalization)
    test_dataset=WeiboDenseDataset(test_data,normalization)
    
    train_loader=torch.utils.data.DataLoader(dataset = train_dataset,
                            batch_size=batch_size,
                            shuffle = True)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=True)
    
    return train_loader,test_loader,train_dataset,test_dataset
    