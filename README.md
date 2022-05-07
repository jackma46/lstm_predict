# UIUC CS598 Deep Learning for Healthcare Final Project
Readmission prediction using deep learning on electronic health records

Article: https://www.sciencedirect.com/science/article/pii/S1532046419301753?via%3Dihub

# Requirements
- Python: https://www.python.org/downloads/
- Pandas: https://pandas.pydata.org/getting_started.html
- Numpy: https://numpy.org/install/
- SciPy: https://scipy.org/install/
- Tensorflow/keras: https://www.tensorflow.org/install/
- Scikit-learn: https://scikit-learn.org/stable/getting_started.html

# Getting Data
This program requires acccess to MIMIC-III Clinical Database through Physionet: https://physionet.org/content/mimiciii/1.4/
Access to this database requires express approval, please follow the instructions.

Once access is granted, download ADMISSIONS, DIAGNOSES_ICD, PROCEDURES_ICD, PRESCRIPTIONS, LABEVENTS, PATIENTS, AND PROCEDUREEVENTS_MV files.
Save these files in csv format to "minicIII" folder in the working directory.

# Constructing Training Data
1. Use stats.py to analyze the data in ADMISSIONS.csv to obtain important numbers such as total number of patients, tootal number of visits, maximum visit per patient, and how many filler rows need to be constructed to fill out the data for training.
2. Use d2v.py to construct the machine driven features. At the top of d2v.py, specify the filepaths to diagnosis, procedure, prescription, and lab codes. Then specify the columns that represent the visit ids and codes in these files. Running d2v.py will produce mdf.csv in the working directory.
3. Use h2v.py to construct the human driven features. At the top of h2v.py, specify the filepaths to admissions, patients, and procedures events. Then specify the maximum number of visit for patient as calculated in stats.py. Running h2v.py will produce hdf.csv in the working directory.
4. As the feature files from MIMIC-III dataset is quite large (should be 480MB and almost 2 million rows of data after adding in fillers), this program is designed to save mdf and hdf separately. Then using combine_features.py, the two sets of feature vectors are combined into the required data structure for training and evaluation.

# Training and Evaluation
1. The input vector file should have 206 columns, where the first column is the patient id, second column is the visit id, fifth column is the 30 day readmission indicator, sixth column is the assigned cost (0 for filler, 3 for readmission, 1 for all others), columns 7 to 21 are human driven features, and columns beyond 21 are machine driven features.
2. Limited changes are made to lstm_predict.py, the training models are exactly the same as in the original repository. 
3. If the columns are different as described in point 1, read_data function should be changed accordingly. To do so, ind1 and ind2 indicates the start and end of feature vectors; pid indicates patient id; adminId and VisitIds both indicate visitId; output indicates the 30 day readmission flag; and Weights is the assigned cost.
4. Output files will be saved directly into the working directory. For organization purposes, these files are moved into the result folder. 
5. Use display_result.py to display the performance indicators similar to shown as table 4 in the paper. At the top of display_result.py, specify the folder where the output files from lstm_predict.py are saved. Running display_result.py will produce table.csv in the working directory.

# Contributions
Originally from https://github.com/caisr-hh/lstm_predict

Modified by Jack Ma for UIUC CS598 Deep Learning for Healthcare

Team 86 - jma46

Paper 288 - Easy

