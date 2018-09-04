# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:43:25 2018

@author: nce3xin
"""

from scipy.io import arff
import pandas as pd

# .xlsx data file path
root="../data/"
origin_pt=root+"origin.xlsx"
train_pt=root+"train.xlsx"
test_pt=root+"test.xlsx"

# .arff data file path
train_arff_pt="../data/train.arff"
test_arff_pt="../data/test.arff"

# read .xlsx file
usecols=[0,2]
train_df=pd.read_excel(train_pt,usecols=usecols)
test_df=pd.read_excel(test_pt,usecols=usecols)
origin_weibo_df=pd.read_excel(origin_pt,sheetname=0)
origin_feature_df=pd.read_excel(origin_pt,sheetname=1)

# read .arff file
train_arff_data = arff.loadarff(train_arff_pt)
train_arff_df = pd.DataFrame(train_arff_data[0])
test_arff_data = arff.loadarff(test_arff_pt)
test_arff_df = pd.DataFrame(test_arff_data[0])

# extract instance serial number
def _extract_serial_number(index):
    index=index[4:]
    return index

def modify_instance_number(df):
    df.rename(columns={'对应weka（训练测试重分后）':'Instance_number'},inplace=True)
    df['Instance_number']=df.iloc[:,0].map(_extract_serial_number)
    df['Instance_number']=df['Instance_number'].astype('int')
    return df

def convert_instance_number_to_int(df):
    df['Instance_number']=df['Instance_number'].astype('int')
    return df
    

train_df=modify_instance_number(train_df)
test_df=modify_instance_number(test_df)

train_arff_df=convert_instance_number_to_int(train_arff_df)
test_arff_df=convert_instance_number_to_int(test_arff_df)

