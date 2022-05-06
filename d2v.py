# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:51:35 2019

@author: aaq109
For queries Contact: awais.ashfaq@hh.se
"""
'''
Modified on May 01, 2022
@author: Jack Ma
For queries Contact: jma46@illinois.edu
'''
from gensim.models import doc2vec
from collections import namedtuple
import scipy.io as sio
import pandas as pd

# Define all the needed inputs
f1 = ("mimicIII/DIAGNOSES_ICD.csv", 2, 4) # file name, visit id column number, code column number
f2 = ('mimicIII/PROCEDURES_ICD.csv', 2, 4)
f3 = ('mimicIII/PRESCRIPTIONS.csv', 2, 10)
f4 = ('mimicIII/LABEVENTS.csv', 2, 3)
patients = 58976 # Found from stats.py
num_filler = 1894863 # Found from stats.py

# Read in the visit, codes and put into a map
admDiagMap = {}
print ("reading in data...")
for f, visit, code in [f1, f2, f3, f4]:
    print ("processing", f)
    infd = open(f, 'r', encoding='ISO-8859-1')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = (tokens[visit])
        d = (tokens[code])
        if (f == 'mimicIII/LABEVENTS.csv'):
            d = "item_" + d        
        diagId = d.replace('"', '')
        if admId in admDiagMap:
            admDiagMap[admId].append(diagId)
        else:
            admDiagMap[admId] = [diagId]
    infd.close()

# Put the codes in order of visit number and find the maximum code per visit
maximum = 0
s1 = []
for key in sorted(admDiagMap):
    value = admDiagMap[key]
    current_len = len(value)
    if current_len > maximum:
        maximum = current_len    
    s1.append(value)
s1.append('')        
print ("The maximum number of codes per visit is", maximum)

# Put the tags/codes into the document
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(s1):
    words = text
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Define the parameters of doc2vec 
Emb_size=185 #size of the visit vector K
window=maximum # Max length of codes in any visit
min_count=0 # Consider all codes
ns=20 # Negative sampling
ns_exponent=-0.75 # Negative because we like to account for rare clinical events
dm=0 # for PV-DBOW

# Get the vectors
model = doc2vec.Doc2Vec(docs, vector_size = Emb_size, window = window, min_count = min_count, workers = 4,negative =ns, ns_exponent=ns_exponent, dm=dm)
d2v=model.dv.vectors

# Add in the fillers to fill out the file.
df = pd.DataFrame(data=d2v[:,:])
filler = df.iloc[patients:,]
filler = pd.concat([filler]*num_filler, ignore_index=True)
df = pd.concat([df, filler], ignore_index=True)

df.to_csv('mdf.csv', index=False)
print("There are", len(df), "total rows of data in mdf.")