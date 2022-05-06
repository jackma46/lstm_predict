'''
Created on May 01, 2022

@author: Jack Ma
For queries Contact: jma46@illinois.edu
'''

import pandas as pd
import numpy as np
from collections import OrderedDict
import collections

df = pd.read_csv('mimicIII/ADMISSIONS.csv')

gs = df.groupby('SUBJECT_ID').size()

patients = len(gs)
visits = len(df)
max_visits = np.amax(gs)
final_rows = len(gs) * np.amax(gs)
num_filler = final_rows - visits - 1
print("There are", patients, "patients in the dataset")
print("There are", visits, "visits in the dataset")
print("The maximum number of visits is", max_visits)
print("There should be", final_rows, "rows in the final data file")
print(num_filler, "rows of filler data needs to be created")

stats = {}

for i in gs:
    if i in stats:
        stats[i] = stats[i] + 1
    else:
        stats[i] = 1 
        
ordered = OrderedDict()
ordered = collections.OrderedDict(sorted(stats.items()))
print("Here are the distribution of (number of visits, patients with this number of visits)")
print(ordered)
