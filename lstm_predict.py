# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 08:51:35 2019

@author: aaq109
"""

import timeit
import numpy as np
from numpy import array
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Read data
N_visits=42 # Maximum number of inpatient visits in the dataset

def read_data(exp, N_visits):
    label='data_lstm_'+str(N_visits)+'.csv'
    print('Reading File: ',label)
    pidAdmMap = {}
    admDetailMap={}
    output=[]
    Weights=[]
    VisitIds=[]
    if exp[0:2]=='11':
        ind1=6
        ind2=206
    elif exp[0:2]=='10':
        ind1=6
        ind2=21
    else:
        ind1=21
        ind2=206
    infd = open (label,'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid=int(float(tokens[0]))
        admId=(tokens[1])
        det=(tokens[ind1:ind2]) #200 if 185 d2v vector is used
        output.append(tokens[4])
        Weights.append(tokens[5])
        VisitIds.append(tokens[1])
        if admId in admDetailMap:
            admDetailMap[admId].append(det)
        else:
            admDetailMap[admId]=det
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid]=[admId]
    infd.close()
    _list = []
    for patient in pidAdmMap.keys():
        a = [admDetailMap[xx] for xx in pidAdmMap[patient]]
        _list.append(a)
    X=np.array([np.array(xi) for xi in _list])
    a,b,c=X.shape
    Y=np.array(output)
    Sample_weight=np.array(Weights)
    X = X.astype(float)
    Y = Y.astype(float)
    Sample_weight = Sample_weight.astype(float)
    Y=Y.reshape(X.shape[0],N_visits,1)
    return X, Y,Sample_weight,VisitIds

def ppv(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    ppv = true_positives / (predicted_positives + K.epsilon())
    return ppv

def npv(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    predicted_negatives = K.sum(K.round(K.clip(1-y_pred, 0, 1)))
    npv = true_negatives / (predicted_negatives + K.epsilon())
    return npv

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train):
    from sklearn.preprocessing import binarize
    from sklearn.metrics import f1_score
    from sklearn.metrics import balanced_accuracy_score,accuracy_score
    import operator
    y_pred = model.predict(X_train).ravel()
    y_test=Y_train.ravel()
    g=Sample_weight_train.ravel()
    g[g==0]=0
    g[g>0]=1
    indices=np.where(g==0)
    y_pred=np.delete(y_pred,indices,0)
    y_test=np.delete(y_test,indices,0)
    score={}
    for thresh in np.arange(0.001,1,0.001):
        y_pred_class=binarize([y_pred],threshold=thresh)[0]
        cm= confusion_matrix(y_test, y_pred_class, labels=[0,1])
        score[thresh]=(48000*cm[1,1]*0.5)-(7000*(cm[1,1]+cm[0,1]))
    thresh=max(score.items(), key=operator.itemgetter(1))[0]

    y_pred = model.predict(X_test).ravel()
    y_test=Y_test.ravel()
    g=Sample_weight_test.ravel()
    g[g==0]=0
    g[g>0]=1
    if exp[2]=='1':
        fpr, tpr, thetas = roc_curve(y_test, y_pred,sample_weight=g,pos_label=1)
        prc, recal, thetas = precision_recall_curve(y_test, y_pred,sample_weight=g)
        indices=np.where(g==0) #Patient gender
        y_pred=np.delete(y_pred,indices,0)
        y_test=np.delete(y_test,indices,0)
    else:
        fpr, tpr, thetas = roc_curve(y_test, y_pred,pos_label=1)
        prc, recal, thetas = precision_recall_curve(y_test, y_pred)

    AUC_test = auc(fpr, tpr)
    PR_auc = auc(recal,prc)


    y_pred=binarize([y_pred],threshold=thresh)[0]
    cm= confusion_matrix(y_test, y_pred, labels=[0,1])
    print(y_test, y_pred, cm)
    cost_saved=(48000*cm[1,1]*0.5)-(7000*(cm[1,1]+cm[0,1]))
    Accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
    Sensitivity_test=cm[1,1]/(cm[1,0]+cm[1,1] + K.epsilon())
    Specificity_test=cm[0,0]/(cm[0,0]+cm[0,1])
    F1_score=f1_score(y_test,y_pred)
    cost_saved=cost_saved/(np.sum(y_test)*(48000-7000)*0.5 + K.epsilon())

    return Accuracy, AUC_test, Sensitivity_test, Specificity_test, PR_auc, F1_score,cost_saved

def save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp):
    label1='AUC_test_'+exp+'.npy'
    label2='Sensitivity_test_'+exp+'.npy'
    label3='Specificity_test_'+exp+'.npy'
    label4='PR_auc_'+exp+'.npy'
    label5='f1_score_'+exp+'.npy'
    label6='cost_saved_'+exp+'.npy'
    np.save(label1, AUC_test)
    np.save(label2, Sensitivity_test)
    np.save(label3, Specificity_test)
    np.save(label4, PR_auc)
    np.save(label5, F1_score)
    np.save(label6, cost_saved)
    val1=np.fromiter(AUC_test.values(), dtype=float)
    val2=np.fromiter(Sensitivity_test.values(), dtype=float)
    val3=np.fromiter(Specificity_test.values(), dtype=float)
    val4=np.fromiter(PR_auc.values(), dtype=float)
    val5=np.fromiter(F1_score.values(), dtype=float)
    val6=np.fromiter(cost_saved.values(), dtype=float)

    print(label1,[np.mean(val1[np.nonzero(val1)]),np.std(val1[np.nonzero(val1)])])
    print(label2,[np.mean(val2[np.nonzero(val2)]),np.std(val2[np.nonzero(val2)])])
    print(label3,[np.mean(val3[np.nonzero(val3)]),np.std(val3[np.nonzero(val3)])])
    print(label4,[np.mean(val4[np.nonzero(val4)]),np.std(val4[np.nonzero(val4)])])
    print(label5,[np.mean(val5[np.nonzero(val5)]),np.std(val5[np.nonzero(val5)])])
    print(label6,[np.mean(val6[np.nonzero(val6)]),np.std(val6[np.nonzero(val6)])])


    return None


## Define different experiments
    # 1111 - HDF+MDF+LSTM+CA
exp='1111'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)
es=EarlyStopping(monitor='val_loss', patience=20, mode='min')

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
  #  print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.3, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)


## Define different experiments
    # 1110 - HDF+MDF+LSTM
exp='1110'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
  #  print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)


## Define different experiments
    # 0111 - MDF+LSTM+CA
exp='0111'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}

#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)
es=EarlyStopping(monitor='val_loss', patience=20, mode='min')

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)

## Define different experiments
    # 0110 - MDF+LSTM
exp='0110'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)

## Define different experiments
    # 1011 - HDF+LSTM+CA
exp='1011'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}

#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)
es=EarlyStopping(monitor='val_loss', patience=20, mode='min')

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)


## Define different experiments
    # 1010 - HDF+LSTM
exp='1010'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}

#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print(model.summary())
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)

## Define different experiments
    # 1101 - HDF+MDF+CA
exp='1101'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))
    model.add(Dense(NN_nodes[1], activation='sigmoid'))
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)

## Define different experiments
    # 1101 - HDF+MDF
exp='1100'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))
    model.add(Dense(NN_nodes[1], activation='sigmoid'))
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.3, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)


## Define different experiments
    # 1001 - HDF+CA
exp='1001'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))
    model.add(Dense(NN_nodes[1], activation='sigmoid'))
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)


## Define different experiments
    # 1000 - HDF only
exp='1000'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))
    model.add(Dense(NN_nodes[1], activation='sigmoid'))
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)

## Define different experiments
    # 1000 - MDF only
exp='0100'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))
    model.add(Dense(NN_nodes[1], activation='sigmoid'))
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)


## Define different experiments
    # 0101 - MDF + CA only
exp='0101'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
F1_score={}
cost_saved={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=80 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10

X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))
    model.add(Dense(NN_nodes[1], activation='sigmoid'))
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print('Training start', 'for iteration ', iter_nm )
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True, validation_split=0.2, callbacks=[es])
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm], F1_score[iter_nm],cost_saved[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test,exp,X_train,Y_train,Sample_weight_train)
    print('Evaluation complete', 'for iteration ', iter_nm )

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc,F1_score,cost_saved, exp)

#print([np.mean(np.fromiter(np.load('cost_saved_1111.npy').item().values(), dtype=float)),np.std(np.fromiter(np.load('cost_saved_1111.npy').item().values(), dtype=float))])

AUC_1111=np.fromiter(np.load('AUC_test_1111.npy', allow_pickle=True).item().values(), dtype=float)
AUC_1110=np.fromiter(np.load('AUC_test_1110.npy', allow_pickle=True).item().values(), dtype=float)
AUC_1011=np.fromiter(np.load('AUC_test_1011.npy', allow_pickle=True).item().values(), dtype=float)
AUC_0111=np.fromiter(np.load('AUC_test_0111.npy', allow_pickle=True).item().values(), dtype=float)
AUC_1101=np.fromiter(np.load('AUC_test_1101.npy', allow_pickle=True).item().values(), dtype=float)

cs_1111=np.fromiter(np.load('cost_saved_1111.npy', allow_pickle=True).item().values(), dtype=float)
cs_1110=np.fromiter(np.load('cost_saved_1110.npy', allow_pickle=True).item().values(), dtype=float)
cs_1011=np.fromiter(np.load('cost_saved_1011.npy', allow_pickle=True).item().values(), dtype=float)
cs_0111=np.fromiter(np.load('cost_saved_0111.npy', allow_pickle=True).item().values(), dtype=float)
cs_1101=np.fromiter(np.load('cost_saved_1101.npy', allow_pickle=True).item().values(), dtype=float)

f1_1111=np.fromiter(np.load('f1_score_1111.npy', allow_pickle=True).item().values(), dtype=float)
f1_1110=np.fromiter(np.load('f1_score_1110.npy', allow_pickle=True).item().values(), dtype=float)
f1_1011=np.fromiter(np.load('f1_score_1011.npy', allow_pickle=True).item().values(), dtype=float)
f1_0111=np.fromiter(np.load('f1_score_0111.npy', allow_pickle=True).item().values(), dtype=float)
f1_1101=np.fromiter(np.load('f1_score_1101.npy', allow_pickle=True).item().values(), dtype=float)


#aucs_mean = [np.mean(AUC_1111), np.mean(AUC_1110)]
#aucs_std = [np.std(AUC_1111), np.std(AUC_1110)]

df_results = pd.DataFrame(np.array([[np.mean(AUC_1111), np.mean(AUC_1110),np.mean(AUC_0111),np.mean(AUC_1011),np.mean(AUC_1101)], \
                      [np.mean(f1_1111), np.mean(f1_1110), np.mean(f1_0111), np.mean(f1_1011), np.mean(f1_1101)], \
                      [np.mean(cs_1111), np.mean(cs_1110),np.mean(cs_0111),np.mean(cs_1011),np.mean(cs_1101)], \
                     ]))
df_std = pd.DataFrame(np.array([[np.std(AUC_1111)/1, np.std(AUC_1110)/1,np.std(AUC_0111)/1,np.std(AUC_1011)/1,np.std(AUC_1101)/1], \
                      [np.std(f1_1111)/1, np.std(f1_1110)/1, np.std(f1_0111)/1, np.std(f1_1011)/1, np.std(f1_1101)/1], \
                      [np.std(cs_1111)/1, np.std(cs_1110)/1, np.std(cs_0111)/1, np.std(cs_1011)/1, np.std(cs_1101)/1], \
                     ]))
df_results.index = ['ROC AUC','F1 score','Cost saved']

df_results.columns = ['Complete Model', 'Without CA', 'Without HDF', 'Without MDF','Without LSTM']
#patterns =  (('/'),('o'))
colors=['blue','skyblue','silver','gray', 'black']
fig, ax = plt.subplots()
#plt.rcParams.update({'figure.figsize': [5, 5], 'font.size': 22})
plt.rcParams.update({'font.size': 20, 'figure.figsize': [10,8]})
ax = df_results[::-1].plot.barh(ax=ax, xerr=np.array(df_std[::-1]).transpose(),color=colors,width=0.7,capsize=5)
ax.legend(bbox_to_anchor=(0.95, 0.30))
#ax.set_xlabel('F1 score / Cost Savings')
plt.tight_layout()
#Options

plt.show()

plt.savefig('fig3.pdf', format='pdf', dpi=1000)


#SIGNIFICANCE TESTS
#label='cost_saved'
#c=np.fromiter(np.load(label+'_1110.npy').item().values(), dtype=float)
#d=np.fromiter(np.load(label+'_1111.npy').item().values(), dtype=float)
#a,b=stats.ttest_ind(c,d)
#print(a,b)

#import scipy.io as sio
#y_pred = model.predict(X_test).ravel()
#sio.savemat('y_pred_new_review2.mat', {'y_pred':y_pred,'Visit_ID':Visit_test.ravel(),'Y_test':Y_test.ravel(),'Sample':Sample_weight_test.ravel()})
