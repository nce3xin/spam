# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:34:49 2018

@author: nce3xin
"""
import sys
sys.path.append("..")

from datetime import datetime
import math
import pandas as pd
import arff
from load_origin_data import origin_feature_df,origin_weibo_df,train_arff_df,train_df,test_arff_df,test_df

def process_df(arff_df,df):
    # arff_df left join df
    # add "user name" column
    merged_arff_df=arff_df.merge(df,how="left",on="Instance_number")
    # add "最远转发时间" and "最近转发时间" column
    merged_arff_df=merged_arff_df.merge(origin_feature_df[['用户ID','最远转发时间','最近转发时间']],how="left",on="用户ID")
    return merged_arff_df
    
def extract_min_max_date(df):
    min_date=min(df['转发时间'])
    max_date=max(df['转发时间'])
    return min_date,max_date

def _format_date(date):
    if isinstance(date,str)==False:                 
        if math.isnan(date):
            return date                     
        
    if '前' in date or '今天' in date:
        return float('nan')
        
    if '月' not in date:
        y=int(date.split(' ')[0].split('-')[0])
        m=int(date.split(' ')[0].split('-')[1])
        d=int(date.split(' ')[0].split('-')[2])
        h=int(date.split(' ')[1].split(':')[0])
        minute=int(date.split(' ')[1].split(':')[1])
        return datetime(y,m,d,h,minute)
    
    year=2018
    month=int(date.split('月')[0])
    day=int(date.split('月')[1].split('日')[0])
    hour=int(date.split('日')[1].strip().split(':')[0])
    minute=int(date.split('日')[1].strip().split(':')[1])
    
    date=datetime(year,month,day,hour,minute)
    return date

def format_date(df):
    df['转发时间']=df['转发时间'].map(_format_date)
    df['转发时间']=df['转发时间'].dropna()

def extract_temporal_ftrs(username,df,min_date,max_date,freq='7D'):
    t=df[df['用户ID']==username]['转发时间']
    t=pd.Series(1,index=t)
    
    min_date=str(min_date)
    max_date=str(max_date)
    
    y1=int(min_date.split(' ')[0].split('-')[0])
    m1=int(min_date.split(' ')[0].split('-')[1])
    d1=int(min_date.split(' ')[0].split('-')[2])
    
    y2=int(max_date.split(' ')[0].split('-')[0])
    m2=int(max_date.split(' ')[0].split('-')[1])
    d2=int(max_date.split(' ')[0].split('-')[2])
    
    start=pd.Series(0,index=[datetime(y1,m1,d1)])
    end=pd.Series(0,index=[datetime(y2,m2,d2)])
    t=start.append(t).append(end)
    
    # downsample the series into several bins according to the given frequency
    # and sum the values of the timestamps falling into a bin.
    t=t.resample(freq).sum()
    
    # nan occurs when nothing falls into this bin. 
    # As the value in the bin represents the number of forwards, so we fill nan with zero.
    t=t.fillna(0)
    
    return t

def build_new_df(usernames,min_date,max_date):
    index=None
    for i,name in enumerate(usernames):
        if i >= 1:
            break
        t=extract_temporal_ftrs(name,origin_weibo_df,min_date,max_date)
        index=t.index
    
    df=pd.DataFrame(columns=['用户ID']+list(index)) # create an empty dataframe
    
    for name in usernames:
        t=extract_temporal_ftrs(name,origin_weibo_df,min_date,max_date)
        d=t.to_dict()
        d['用户ID']=name
        df.loc[df.shape[0]+1] = d # add a new line to new dataframe
    return df
    
def save_to_arff(df,out_pt):
    arff.dump(out_pt,df.values.tolist(),relation='temporal features about weibo forward times',names=list(df.columns))

def move_label_to_last_column(df):
    cols=list(df.columns.values)
    cols=cols[:13]+cols[14:]+[cols[13]]
    df=df[cols]
    return df

# process train and test dataset
merged_train_df=process_df(train_arff_df,train_df)
merged_train_df['用户ID'].fillna('Dummy',inplace=True) # fillna because there are virtual users in train dataset
merged_test_df=process_df(test_arff_df,test_df)

# format origin_weibo_df "转发时间" column
if str(origin_weibo_df['转发时间'].dtype)!='datetime64[ns]':
    format_date(origin_weibo_df)
    
# extract min max date from overall data (train and test data as a whole)
# min_date,max_date=extract_min_max_date(origin_feature_df)
min_date,max_date=extract_min_max_date(origin_weibo_df)

# build new dataframe consisting of 
# all usernames and their temporal features about forward times information in train and test set.
new_df=build_new_df(pd.concat([merged_train_df.iloc[:766,:],merged_test_df],ignore_index=True)['用户ID'],min_date,max_date)


merged_train_df=merged_train_df.merge(new_df,how="left",on="用户ID")
merged_train_df.fillna(0,inplace=True)
merged_test_df=merged_test_df.merge(new_df,how="left",on="用户ID")

del merged_train_df['最远转发时间']
del merged_train_df['最近转发时间']
del merged_train_df['用户ID']
del merged_test_df['最远转发时间']
del merged_test_df['最近转发时间']
del merged_test_df['用户ID']

# move label to last column
merged_train_df=move_label_to_last_column(merged_train_df)
merged_test_df=move_label_to_last_column(merged_test_df)

# save to file
train_out_pt='../data/gen/train.csv'
test_out_pt='../data/gen/test.csv'

merged_train_df.to_csv(train_out_pt)
merged_test_df.to_csv(test_out_pt)