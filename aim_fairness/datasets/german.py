# German Dataset
# Preprocessing is large based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/german.py
# 
#   Features                :   57  Dimension ( / {credit, sex})
#   Sensitive Attribute     :    1  Dimension (sex)
#   Ground Truth Label      :    1  Dimension (credit)
#
#   Please note that the features does not contain the sensitive attribute. 
#   This is in contrast to the assumptions of Ruoss et. al https://github.com/eth-sri/lcifr/issues/1
#   which are by the way based on Data Producer - Data Consumer Framework of McNamara et. al
#
#   We highly suggest that you concatenate the features and sensitive instead in cases where 
#   Data Producer is assumed to have access to every feature, including the sensitive attribute

import os
import urllib.request as ur

import torch
import pandas as pd
import numpy as np


# External File Info
german_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/'
german_download_folder = "data/"
german_filename = 'german.data'

# Column Names
column_names = [
    'status', 'months', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
    'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
    'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone',
    'foreign_worker', 'credit'
]

# Personal Status Map
personal_status_map = {'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'}

# Sensitive Attribute
sensitive_attribute = 'sex'

def _download_one( filename ):
    """
    Download a file if not present
    Default save path is "data/" folder
    """

    filepath = german_download_folder + filename

    if not os.path.exists( german_download_folder ):
        os.makedirs( german_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( german_url + filename, filepath )
    else: 
        print( "Found and verified ", filepath )

def _german_download():
    _download_one( german_filename )

def _read_data_file(path, device, normalize=True):
    # Load data
    data_frame = pd.read_csv(path, sep=' ', header=None, names=column_names)

    # Personal Status -> Sex
    data_frame['sex'] = data_frame['personal_status'].replace(personal_status_map)
    data_frame.drop('personal_status', axis=1, inplace=True)

    # Divide the dataset
    features, labels, sensitive = data_frame.drop(['credit', sensitive_attribute], axis=1), data_frame['credit'], data_frame[sensitive_attribute]
    
    # Preprocess Sensitive Attribute
    a = np.logical_not( pd.Categorical( sensitive ).codes )

    # Categorize where a column is a (1) continuous variable or a (2) categorical variable
    # We need to store the column names because their index will change once we convert categorical variables to one-hot encodings
    continuous_vars = []                                        
    categorical_columns = []                                    
    for col in features.columns:
        if (features[col].isnull().sum() > 0):
            features.drop(col, axis=1, inplace=True)
        else:
            if (features[col].dtype == np.object):
                categorical_columns += [col]
            else:
                continuous_vars += [col]

    # Transform Categorical Variables into their One-hot Encoding
    features = pd.get_dummies(features, columns=categorical_columns, prefix_sep='=')
    continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

    # This essentially stores the column name of the categorical variables and the list of indexes of their one-hot representations, 
    # e.g. "wordclass" = [6,7,8,9,10,11,12] because workclass=1, workclass=2 etc
    # Notice the 'sex' column, which is also our sensitive variable, has only 2 unique values
    one_hot_columns = {}
    for column_name in categorical_columns:
        ids = [i for i, col in enumerate(features.columns) if col.startswith("{}=".format(column_name))]
        if (len(ids) > 0):
            assert len(ids) == ids[-1] - ids[0] + 1
        one_hot_columns[column_name] = ids

    # Convert data to torch tensor
    x = torch.tensor( features.values.astype(np.float32), device=device )
    y = 2 - torch.tensor( labels.values.astype(np.int64), device=device )   # We need to subtract from 2 because the resulting tensor have values {1,2}
    a = torch.tensor( a.astype(np.bool), device=device ) * 1

    # Normalize the values of the continuous variables
    if (normalize):
        columns = continuous_columns if continuous_columns is not None else np.arange(x.shape[1])
        mean, std = x.mean(dim=0)[columns], x.std(dim=0)[columns]
        x[:, columns] = (x[:, columns] - mean) / std

    return x, y, a

def load_dataset(download=True, device="cpu"):
    device = device
    if (device == None):
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    if (download):
        _german_download()

    features, label, sensitive = _read_data_file( german_download_folder + german_filename, device=device )

    return features, label, sensitive