# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:18:25 2018

@author: nce3xin
"""
from sklearn import metrics

# true label and predicted label file
file_pt='../testset_results/GRU.csv'

# TPR equals recall
def TPR(category):
    return recall(category)

def _pick_true_negative(category):
    with open(file_pt,'r') as f:
        lines=[line for line in f.readlines()[1:] if line.strip().split(',')[0]!=str(category)]
    return lines

def FPR(category):
    lines=_pick_true_negative(category)
    true=[line.strip().split(',')[0] for line in lines]
    pred=[line.strip().split(',')[1] for line in lines]
    assert len(true)==len(pred)
    correct=0
    for i in range(len(true)):
        if pred[i]==str(category) and true[i]!=pred[i]:
            correct+=1
    total=len(lines)
    return correct/total

def _pick_true_positive(category):
    with open(file_pt,'r') as f:
        # [1:] in order to skip header
        lines=[line for line in f.readlines()[1:] if line.strip().split(',')[0]==str(category)]
    return lines

def recall(category):
    lines=_pick_true_positive(category)
    true=[line.strip().split(',')[0] for line in lines]
    pred=[line.strip().split(',')[1] for line in lines]
    correct=0
    assert len(true)==len(pred)
    for i in range(len(true)):
        if true[i]==pred[i]:
            correct+=1
    total=len(lines)
    return correct/total

def _pick_pred_positive(category):
    with open(file_pt,'r') as f:
        lines=[line for line in f.readlines()[1:] if line.strip().split(',')[1]==str(category)]
    return lines

def precision(category):
    lines=_pick_pred_positive(category)
    true=[line.strip().split(',')[0] for line in lines]
    correct=0
    for i in range(len(true)):
        if true[i]==str(category):
            correct+=1
    total=len(lines)
    return correct/total

def F1(precision,recall):
    f1=2*precision*recall/(precision+recall)
    return f1
    
def AUC(category):
    with open(file_pt,'r') as f:
        context=[line.strip() for line in f.readlines()[1:]]
    lines=[]
    for line in context:
        true=int(line.split(',')[0])
        pred=int(line.split(',')[1])
        lines.append([true,pred])
    for line in lines:
        if line[0]==category and line[0]==line[1]:
            line[0]=line[1]=1
        elif line[0]==category and line[0]!=line[1]:
            line[0]=1
            line[1]=0
        elif line[0]!=category and line[0]==line[1]:
            line[0]=line[1]=0
        elif line[0]!=category and line[0]!=line[1]:
            line[0]=0
            line[1]=1
    true=[line[0] for line in lines]
    pred=[line[1] for line in lines]
    
    auc=metrics.roc_auc_score(true,pred)
    
    return auc

def accuracy():
    with open(file_pt,'r') as f:
        lines=[line.strip() for line in f.readlines()[1:]]
        true=[line.split(',')[0] for line in lines]
        pred=[line.split(',')[1] for line in lines]
        assert len(true)==len(pred)
        correct=0
        for i in range(len(true)):
            if true[i]==pred[i]:
                correct+=1
    total=len(true)
    return correct/total

acc=accuracy()
print('acc={:.3f}'.format(acc))
print()

for category in range(3):
    print('for category {}:'.format(category))
    print('-'*20)
    tpr=TPR(category)
    fpr=FPR(category)
    p=precision(category)
    r=recall(category)
    f1=F1(p,r)
    auc=AUC(category)
    print('TPR={:.3f}'.format(tpr))
    print('FPR={:.3f}'.format(fpr))
    print('precision={:.3f}'.format(p))
    print('recall={:.3f}'.format(r))
    print('f1={:.3f}'.format(f1))
    print('auc={:.3f}'.format(auc))