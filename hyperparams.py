# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:17:24 2018

@author: nce3xin
"""

seed_num=1

learning_rate=1e-3
#epochs=109
#epochs=90
epochs=20
batch_size=16

log_interval=1
no_cuda=False

MODEL='LSTM'
cnn_out_dims=25

CNN_mapping=False

normalization=False
standard_scale=False
min_max_scaler=False