'''
Created on May 01, 2022

@author: Jack Ma
For queries Contact: jma46@illinois.edu
'''

import pandas as pd

print("Reading in the data")
hdf = pd.read_csv('hdf.csv', index_col=False)
mdf = pd.read_csv('mdf.csv', index_col=False)
print("Combining the data")
result = pd.concat([hdf, mdf], axis=1)

# Add weights and sort by patient Id 
result = result.sort_values(['SUBJECT_ID', 'HADM_ID'])   

print("Writing data to file, this might take a while...")
result.to_csv('data_lstm_42.csv', index=False)