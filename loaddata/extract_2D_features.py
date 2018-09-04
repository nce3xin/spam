# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:46:02 2018

@author: nce3xin
"""
import torch 
import pandas as pd

def extract_min_max_forward_times(df):
    min_value=df.iloc[:,13:-1].min().min()
    max_value=df.iloc[:,13:-1].max().max()
    return min_value,max_value

# ts only is a vector including only time series (not including instance number or usernames etc.)
def to2D(ts,min_forward_times,max_forward_times):
    tensor=torch.zeros((max_forward_times-min_forward_times+1,len(ts)),dtype=torch.long)
    for i,val in enumerate(ts):
        val=int(val)
        tensor[val,i]=1
    return tensor
    
def convertTo2D(df,min_forward_times,max_forward_times):
    n_row=len(df)
    for i in range(n_row):
        ts=df.iloc[i,13:-1]
        
    data=torch.zeros(n_row,max_forward_times-min_forward_times+1,len(ts),dtype=torch.long)
        
    for i in range(n_row):
        ts=df.iloc[i,13:-1]
        tensor2D=to2D(ts,min_forward_times,max_forward_times)
        data[i]=tensor2D
    return data
        

def extract_2D_features():
    # load merged_train_df and merged_test_df
    merged_train_df=pd.read_csv('data/gen/train.csv')
    merged_test_df=pd.read_csv('data/gen/test.csv')
    
    # convert temporal part to 2D
    min_forward_times,max_forward_times=extract_min_max_forward_times(pd.concat([merged_train_df,merged_test_df],ignore_index=True))
    
    train_2d_features=convertTo2D(merged_train_df,min_forward_times,max_forward_times)
    test_2d_features=convertTo2D(merged_test_df,min_forward_times,max_forward_times)
    
    return train_2d_features,test_2d_features

def extract_labels_for_2D_features():
    merged_train_df=pd.read_csv('data/gen/train.csv')
    merged_test_df=pd.read_csv('data/gen/test.csv')
    train_labels_for_2D_ftrs=merged_train_df['Class']
    test_labels_for_2D_ftrs=merged_test_df['Class']
    
    def convert_label_to_category(series):
        series=series.astype('category')
        series=series.cat.rename_categories([0,1,2]).astype(int)
        return series
    
    train_labels_for_2D_ftrs=convert_label_to_category(train_labels_for_2D_ftrs)
    test_labels_for_2D_ftrs=convert_label_to_category(test_labels_for_2D_ftrs)
    
    train_labels_for_2D_ftrs=torch.from_numpy(train_labels_for_2D_ftrs.values)
    test_labels_for_2D_ftrs=torch.from_numpy(test_labels_for_2D_ftrs.values)
    
    train_labels_for_2D_ftrs=train_labels_for_2D_ftrs.unsqueeze(0)
    test_labels_for_2D_ftrs=test_labels_for_2D_ftrs.unsqueeze(0)
    
    train_labels_for_2D_ftrs=train_labels_for_2D_ftrs.long()
    test_labels_for_2D_ftrs=test_labels_for_2D_ftrs.long()
    
    return train_labels_for_2D_ftrs,test_labels_for_2D_ftrs