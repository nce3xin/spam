# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:38:47 2018

@author: nce3xin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

model_name='LSTM'
file_name=model_name+'_40-60epoch.xlsx'

# plot train accuracy vs epoch
train_acc_df=pd.read_excel(file_name,sheet_name=0)
plt.figure(figsize=(12,8))
sns.pointplot(train_acc_df['epoch'].values, train_acc_df['train accuracy'].values, alpha=0.8,
              color='b')
plt.ylabel('train accuracy', fontsize=12)
plt.xlabel('epoch', fontsize=12)
axes=plt.gca()
axes.set_ylim([0.8,1])
plt.savefig(model_name+'/'+'train_acc_vs_epoch.png',dpi=300)
plt.show()

# plot test accuracy vs epoch
test_acc_df=pd.read_excel(file_name,sheet_name=1)
plt.figure(figsize=(12,8))
sns.pointplot(train_acc_df['epoch'].values, test_acc_df['test accuracy'].values, alpha=0.8,
              color='b')
plt.ylabel('test accuracy', fontsize=12)
plt.xlabel('epoch', fontsize=12)
axes=plt.gca()
axes.set_ylim([0.5,1])
plt.savefig(model_name+'/'+'test_acc_vs_epoch.png',dpi=300)
plt.show()

