'''
Created on May 01, 2022

@author: Jack Ma
For queries Contact: jma46@illinois.edu
'''

from collections import OrderedDict
import pandas as pd
import datetime
import numpy as np

admissions = pd.read_csv('mimicIII/ADMISSIONS.csv')
patients = pd.read_csv('mimicIII/PATIENTS.csv')
procedures = pd.read_csv('mimicIII/PROCEDUREEVENTS_MV.csv')
max_visits = 42 # This comes from the stats.py

# Find the total procedues by patient
procedures['COUNT'] = 1
procedures = procedures[['SUBJECT_ID', 'COUNT']]
procedures = procedures.groupby(['SUBJECT_ID']).sum()
procedures = procedures.reset_index()

# Combine admissions and patient data
cleanedAdmission = admissions.drop(['ROW_ID' , 'DEATHTIME', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'EDREGTIME', 'EDOUTTIME', 'HAS_CHARTEVENTS_DATA', 'HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS'], axis=1)
cleanedPatients = patients.drop(['ROW_ID',	'DOD',	'DOD_HOSP',	'DOD_SSN',	'EXPIRE_FLAG'], axis=1)
combined = cleanedAdmission.set_index('SUBJECT_ID').join(cleanedPatients.set_index('SUBJECT_ID'))
combined['ADMITTIME'] = pd.to_datetime(combined['ADMITTIME'])
combined['DISCHTIME'] = pd.to_datetime(combined['DISCHTIME']) 
combined['DOB'] = pd.to_datetime(combined['DOB'].str.slice(0,10))

# DOB before 1915 cause overflow error when calculating age
mask = combined['DOB'].lt(datetime.datetime(1915, 1, 1))
combined.loc[mask, 'DOB'] = datetime.datetime(1915, 1, 1)

# Find the total duration for each patient
combined['DURATION'] = (combined['DISCHTIME'] - combined['ADMITTIME']).astype('timedelta64[h]')
combined['AGE'] = (combined['ADMITTIME'] - combined['DOB']).astype('timedelta64[h]')
combined = combined.sort_values(['SUBJECT_ID', 'ADMITTIME'])
totalDuration = combined.groupby(['SUBJECT_ID']).sum()
totalDuration = totalDuration.reset_index()
combined = combined.reset_index()

# Calculate 30 day and 90 day readmission and put in TLOS and number of procedures
combined['FLAG_30'] = 0
combined['FLAG_90'] = 0
combined['TLOS'] = 0.0
combined['PROCEDURE_COUNT'] = 0
previousPatient = None
previousDischarge = None
currentVisit = 0
for i in range(0, len(combined)):
    currentPatient = combined.loc[i, 'SUBJECT_ID']
    if currentPatient == previousPatient:
        currentVisit = currentVisit + 1
        previousDischarge = combined.loc[i-1, 'DISCHTIME']
        currentAdmission = combined.loc[i, 'ADMITTIME']
        diff = (currentAdmission - previousDischarge).total_seconds()
        if (diff < 2592000.0):
            combined.loc[i, 'FLAG_30'] = 1
        if (diff < 7776000.0):
            combined.loc[i, 'FLAG_90'] = 1
    else:
        currentVisit = 0    
    
    tlos = totalDuration.loc[totalDuration['SUBJECT_ID'] == currentPatient, 'DURATION'].to_numpy()
    procedure_count = procedures.loc[procedures['SUBJECT_ID'] == currentPatient, 'COUNT'].to_numpy()
    combined.loc[i, 'TLOS'] = tlos[0]
    if procedure_count.size != 0: 
        combined.loc[i, 'PROCEDURE_COUNT'] = procedure_count[0]
    previousPatient = currentPatient
    
# One-hot encoding of gender
combined['MALE'] = combined.apply(lambda row: 1 if row['GENDER'] == 'M' else 0, axis=1)
combined['FEMALE'] = combined.apply(lambda row: 1 if row['GENDER'] == 'F' else 0, axis=1)

# One-Hot encoding of admission type   
combined['NEWBORN'] = combined.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'NEWBORN' else 0, axis=1)
combined['ELECTIVE'] = combined.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'ELECTIVE' else 0, axis=1)
combined['EMERGENCY'] = combined.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'EMERGENCY' else 0, axis=1)
combined['URGENT'] = combined.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'URGENT' else 0, axis=1)

# One-Hot encoding of ethnicity
combined['ASIAN'] = combined.apply(lambda row: 1 if 'ASIAN' in row['ETHNICITY'] else 0, axis=1)
combined['BLACK'] = combined.apply(lambda row: 1 if 'BLACK' in row['ETHNICITY'] else 0, axis=1)
combined['LATINO_HISPANIC'] = combined.apply(lambda row: 1 if 'LATINO' in row['ETHNICITY'] else 0, axis=1)
combined['MIDDLE_EASTERN'] = combined.apply(lambda row: 1 if 'MIDDLE' in row['ETHNICITY'] else 0, axis=1)
combined['NATIVE'] = combined.apply(lambda row: 1 if 'NATIVE' in row['ETHNICITY'] else 0, axis=1)
combined['WHITE'] = combined.apply(lambda row: 1 if ('WHITE' in row['ETHNICITY'] or 'PORTUGUESE' in row['ETHNICITY']) else 0, axis=1)   

# Add in weights
combined['WEIGHTS'] = combined.apply(lambda row: 3 if row['FLAG_30'] == 1 else 1, axis=1)

combined = combined.drop(['ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE', 'ETHNICITY', 'DOB', 'GENDER'], axis=1)
combined = combined.loc[:, ['SUBJECT_ID','HADM_ID','DURATION','FLAG_90', 'FLAG_30', 'WEIGHTS', 'MALE', 'FEMALE', 'NEWBORN', 'ELECTIVE', 'EMERGENCY', 'URGENT', 'ASIAN', 'BLACK', 'LATINO_HISPANIC', 'MIDDLE_EASTERN', 'NATIVE', 'WHITE', 'AGE', 'TLOS', 'PROCEDURE_COUNT']]

# Fill data so that each patient has
patient_counts = combined.groupby('SUBJECT_ID')['SUBJECT_ID'].count()
new_visit_id = 200000
for a, b in patient_counts.items():
    if b < max_visits:
        new_rows = np.zeros([max_visits - b, 21])
        new_rows[:,0] = a
        for j in range(len(new_rows)):
            new_rows[j,1] = new_visit_id
            new_visit_id = new_visit_id + 1
        rows = pd.DataFrame(new_rows, columns=combined.columns)
        combined = pd.concat([combined, rows], ignore_index=True)
        if (a % 1001 == 0):
            print('processing... ', a)

# Clean and output the data
combined = combined.sort_values(['HADM_ID'])            
combined.to_csv('hdf.csv', index=False)
print ("There are", len(combined), "total rows of data in hdf.")
