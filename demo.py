# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:29:45 2018

@author: nce3xin
"""

import hyperparams
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from loaddata.load_dataset import load_2d_data_for_CNN
from loaddata.load_dataset import load_cnn_dense,load_temporal_data
from models import model_MLP,model_two_stage,model_RNN_LSTM_GRU,model_CNN
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

verbose=True
torch.manual_seed(hyperparams.seed_num)

def is_rnn_lstm_gru(model_name):
    res = (model_name=='RNN' or model_name=='LSTM' or model_name=='GRU')
    return res

def train(model, device, train_loader, optimizer, epoch,criterion,model_name):
    model.train()
    correct=0
    running_loss=0
    losses=[]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.target=data.float(),target.float()
        optimizer.zero_grad()
        if is_rnn_lstm_gru(model_name):
            #output = model(data.unsqueeze(-1),hyperparams.batch_size)
            bs=target.size()[0] # bs means batch size
            output = model(data.unsqueeze(-1),bs)
        elif model_name=='CNN':
            output,_=model(data)
        else:
            output = model(data)
        target=target.long()
        loss = criterion(output, target.squeeze())
        running_loss+=loss.item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if verbose:
            if batch_idx % hyperparams.log_interval == 0:
                cur_loss=running_loss / hyperparams.log_interval
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), cur_loss))
                losses.append(cur_loss)
                running_loss=0
    acc=100. * correct / len(train_loader.dataset)
    if verbose:
        print('Train accuracy: {}/{} ({:.4f}%)\n'.format(correct,len(train_loader.dataset),acc))
    #return epoch,acc,running_loss/len(train_loader)
    return epoch,acc,losses
    
def test(model, device, test_loader,model_name):
    model.eval()
    test_loss = 0
    correct = 0
    predicted_labels=[]
    targets=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data.target=data.float(),target.float()
            if is_rnn_lstm_gru(model_name):
                bs=target.size()[0] # bs means batch size
                output = model(data.unsqueeze(-1),bs)
            elif model_name=='CNN':
                output,_=model(data)
            else:
                output=model(data)
            target=target.long()
            #test_loss += F.nll_loss(output, target.squeeze(), size_average=False).item() # sum up batch loss
            test_loss += data.size()[0] * F.nll_loss(output, target.squeeze()).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predicted_labels+=(pred.cpu().squeeze().numpy().tolist())
            targets+=(target.cpu().squeeze().numpy().tolist())
            
    test_loss /= len(test_loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc=100. * correct / len(test_loader.dataset)
    return acc,predicted_labels,targets

def save_model(model,pt):
    print('save model to: ',pt)
    torch.save(model.state_dict(),pt)
    print('saving model done.')

def save_losses_figure(save_pt,losses,title):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(save_pt+title+'.jpg',dpi=300)
    
def writeToFile(out_pt,dataset):
    data=pd.DataFrame(dataset.ftrs)
    data.to_csv(out_pt)
    
'''
train_pt='data/train.arff'
test_pt='data/test.arff'
'''
train_pt='data/gen/train.csv'
test_pt='data/gen/test.csv'
#train_loader,test_loader,train_dataset,test_dataset=load_data(train_pt,test_pt)
#train_loader,test_loader,train_dataset,test_dataset=load_temporal_data(train_pt,test_pt)

MODEL=hyperparams.MODEL

if MODEL=='CNN':
    train_loader,test_loader,train_dataset,test_dataset=load_2d_data_for_CNN()
else:
    if hyperparams.CNN_mapping:
        train_loader,test_loader,train_dataset,test_dataset=load_cnn_dense()
        #writeToFile('data/ftrs/'+MODEL+'_ftrs.csv',train_loader.dataset)
        writeToFile('data/ftrs/train/'+'CNN_outdims='+str(hyperparams.cnn_out_dims)+'_&origin'+'_ftrs.csv',train_loader.dataset)
        writeToFile('data/ftrs/test/'+'CNN_outdims='+str(hyperparams.cnn_out_dims)+'_&origin'+'_ftrs.csv',test_loader.dataset)
    else:
        train_loader,test_loader,train_dataset,test_dataset=load_temporal_data('data/gen/train.csv','data/gen/test.csv',hyperparams.normalization)

def main():

    torch.manual_seed(hyperparams.seed_num)

    use_cuda = not hyperparams.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    
    n_class=train_dataset.get_n_class()
    in_ftrs=train_dataset.get_in_ftrs()
    
    if MODEL=='MLP':
        model = model_MLP.Net(in_ftrs,n_class).to(device)
        criterion=nn.NLLLoss()
    elif MODEL=='TWO_STAGE':
        model=model_two_stage.Model().to(device)
        criterion=nn.CrossEntropyLoss()
    elif MODEL=='RNN':
        model=model_RNN_LSTM_GRU.RNNModel(1,100,3,1,"RNN").to(device)
        criterion=nn.CrossEntropyLoss()
    elif MODEL=='LSTM':
        model=model_RNN_LSTM_GRU.LSTMModel(1,100,3,1,"LSTM").to(device)
        criterion=nn.CrossEntropyLoss()
    elif MODEL=='GRU':
        model=model_RNN_LSTM_GRU.GRUModel(1,100,3,1,"GRU").to(device)
        criterion=nn.CrossEntropyLoss()
    elif MODEL=='CNN':
        model=model_CNN.CNNModel().to(device)
        criterion=nn.NLLLoss()
    
    model=model.float()
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    
    best_test_acc=0
    best_epoch=0
    
    since=time.time()
    losses=[]
    for epoch in range(1, hyperparams.epochs + 1):
        e,train_acc,loss=train(model, device, train_loader, optimizer, epoch,criterion,MODEL)
        #losses.append(loss)
        losses+=loss # attention, here loss is list
        acc,predicted_labels,targets=test(model, device, test_loader,MODEL)
        if acc>best_test_acc:
            best_test_acc=acc
            best_epoch=e
    time_elapsed=time.time()-since
    print('best test accuracy: {:.4f}% in epoch {}'.format(best_test_acc,best_epoch))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    
    # save losses figures
    if MODEL=='CNN':
        save_losses_figure('losses_figures/',losses,MODEL+'_epoch='+str(hyperparams.epochs)+'_outdims='+str(hyperparams.cnn_out_dims))
    elif not hyperparams.CNN_mapping:
        save_losses_figure('losses_figures/',losses,MODEL+'_epoch='+str(hyperparams.epochs)+'noCNNmapping')
    else:
        save_losses_figure('losses_figures/',losses,MODEL+'_epoch='+str(hyperparams.epochs)+'_outdims='+str(hyperparams.cnn_out_dims))
    
    file_pt='testset_results/'+MODEL+'.csv'
    with open(file_pt,'w') as f:
        assert len(targets)==len(predicted_labels)
        f.write("true label,predicted label\n")
        for i in range(len(targets)):
            f.write(str(targets[i])+","+str(predicted_labels[i])+'\n')
    

    # save model
    if MODEL=='CNN':
        save_model_pt='models_saved/'+MODEL+"_"+str(device)+'_epoch='+str(hyperparams.epochs)+'_outdims='+str(hyperparams.cnn_out_dims)+'.pth'
        save_model(model,save_model_pt)
    
    return losses
    
losses=main()