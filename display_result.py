'''
Created on May 01, 2022

@author: Jack Ma
For queries Contact: jma46@illinois.edu
'''

import numpy as np
import pandas as pd

filepath = "result"

desc_0100 = "MDF"
desc_0101 = "MDF+CA"
desc_0110 = "MDF+LSTM"
desc_0111 = "MDF+LSTM+CA"
desc_1000 = "HDF"
desc_1001 = "HDF+CA"
desc_1010 = "HDF+LSTM"
desc_1011 = "HDF+LSTM+CA"
desc_1100 = "HDF+MDF"
desc_1101 = "HDF+MDF+CA"
desc_1110 = "HDF+MDF+LSTM"
desc_1111 = "HDF+MDF+LSTM+CA"
desc = [desc_0100, desc_0101, desc_0110, desc_0111,
        desc_1000, desc_1001, desc_1010, desc_1011,
        desc_1100, desc_1101, desc_1110, desc_1111]
all = ["0100", "0101", "0110", "0111", 
       "1000", "1001", "1010", "1011",
       "1100", "1101", "1110", "1111"]
all_data=np.zeros([12,7])
for i, name in enumerate(all):
    auc = np.fromiter(np.load(filepath+'/AUC_test_'+name+'.npy', allow_pickle=True).item().values(), dtype=float)
    auc_mean = np.mean(auc)
    auc_std = np.std(auc)
    cs = cs_1111=np.fromiter(np.load(filepath+'/cost_saved_'+name+'.npy', allow_pickle=True).item().values(), dtype=float)
    cs_mean = np.mean(cs)
    cs_std = np.std(cs)
    f1=np.fromiter(np.load(filepath+'/f1_score_'+name+'.npy', allow_pickle=True).item().values(), dtype=float)
    f1_mean = np.mean(f1)
    f1_std = np.std(f1)
    all_data[i][0]=name
    all_data[i][1]=auc_mean
    all_data[i][2]=auc_std
    all_data[i][3]=f1_mean
    all_data[i][4]=f1_std
    all_data[i][5]=cs_mean
    all_data[i][6]=cs_std
    
df = pd.DataFrame(all_data, columns=['name', 'ROC-AUC', 'RA_STD', 'F1_Score', 'F1_STD', 'Cost_Savings', 'CS_STD'])
df2 = pd.DataFrame(desc, columns=['desc'])
df = pd.concat([df, df2], axis=1)

df['HDF'] = df.apply(lambda row: 1 if 'HDF' in row['desc'] else 0, axis=1)
df['MDF'] = df.apply(lambda row: 1 if 'MDF' in row['desc'] else 0, axis=1)
df['LSTM'] = df.apply(lambda row: 1 if 'LSTM' in row['desc'] else 0, axis=1)
df['CA'] = df.apply(lambda row: 1 if 'CA' in row['desc'] else 0, axis=1)

df = df.drop(['name', 'desc'], axis=1)
df = df.loc[:, ['HDF','MDF','LSTM','CA', 'ROC-AUC', 'RA_STD', 'F1_Score', 'F1_STD', 'Cost_Savings', 'CS_STD']]

df.to_csv('table.csv', index=False)
