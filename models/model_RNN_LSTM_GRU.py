# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:07:58 2018

@author: nce3xin
"""

import torch
import torch.nn as nn

#use_cuda = not hyperparams.no_cuda and torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum

        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)

# standard RNN model
class  RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    def forward(self, x, batchSize):
        h0 = torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum).to(device)
        x=x.float()
        h0=h0.float()
        rnnOutput, hn = self.cell(x, h0) # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batchSize, self.hiddenNum).to(device)
        fcOutput = self.fc(hn)
        fcOutput=fcOutput.to(device)

        return fcOutput
    
# LSTM model
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    # if batch_first is true, then the input and output tensors are provided as (batch,seq,feature)
    # else input of shape (seq_len, batch, input_size)
    # h_0,h_n of shape (num_layers*num_directions,batch,hidden_size)
    # c_0,c_n of shape (num_layers*num_directions,batch,hidden_size)
    def forward(self, x, batchSize):

        h0 = torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum).to(device)
        c0 = torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum).to(device)
        
        x=x.float()
        h0=h0.float()
        c0=c0.float()
        
        
        rnnOutput, hn = self.cell(x, (h0, c0))  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn[0].view(batchSize, self.hiddenNum).to(device)
        fcOutput = self.fc(hn)

        return fcOutput
    
# GRU model
class GRUModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    # if batch_first is true, then the input and output tensors are provided as (batch,seq,feature)
    # else input of shape (seq_len, batch, input_size)
    # h_0,h_n of shape (num_layers*num_directions,batch,hidden_size)
    def forward(self, x, batchSize):

        h0 = torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum).to(device)
        x=x.float()
        h0=h0.float()
        rnnOutput, hn = self.cell(x, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batchSize, self.hiddenNum).to(device)
        fcOutput = self.fc(hn)

        return fcOutput