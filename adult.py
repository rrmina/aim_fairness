# Adult dataset
# Preprocessing is largely based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/adult.py
# 
#   Features              : 102 Dimensions ( / {Sex, Income})
#   Sensitive attribute   :   1 Dimension  (Sex)
#   Ground Truth Label    :   1 Dimension  (Income)
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
adult_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
adult_download_folder = "data/"

# File Names 
adult_filenames = {
    "train": "adult.data",
    "test": "adult.test"
}

# Column Names
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Sensitive Attribute
sensitive_attribute = 'sex'

def _download_one( filename ):
    """
    Download a file if not present
    Default save path is "data/" folder
    """

    filepath = adult_download_folder + filename

    if not os.path.exists( adult_download_folder ):
        os.makedirs( adult_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( adult_url + filename, filepath )
    else: 
        print( "Found and verified ", filepath )

def _adult_download():
    for key in adult_filenames:
        _download_one(adult_filenames[key])

def _read_data_file(path, device, train=True, normalize=True):

    # Load CSV
    if (train):
        data_frame = pd.read_csv(path, sep=',', header=None, names=column_names)            
    else:
        data_frame = pd.read_csv(path, sep=',', header=0, names=column_names)               # First line contains a random comment  
    
    # Basic dataframe preprocessing
    data_frame = data_frame.applymap(lambda x: x.strip() if isinstance(x, str) else x)      # Preprocess string variables
    data_frame.replace(to_replace='?', value=np.nan, inplace=True)                          # Replace '?' to NaN
    data_frame.dropna(axis=0, inplace=True)                                                 # Drop missing values

    # Encode labels
    label_map = ( {'<=50K': 0, '>50K': 1} if train else {'<=50K.': 0, '>50K.': 1} )
    data_frame.replace(label_map, inplace=True)                                             # Replace labels from string to Binary

    # Split Features and labels
    features, labels, sensitive = data_frame.drop(['income', sensitive_attribute], axis=1), data_frame['income'], data_frame[sensitive_attribute]

    # Preprocess sensitive attribute
    # Do this before converting categorical to one-hot though pd.get_dummies
    # Otherwise you will encounter a KeyError, because the 'sex' column is already gone -> 'sex0', 'sex1'
    a = np.logical_not( pd.Categorical( sensitive ).codes )             # pd.Categorical().codes converts categorical strings to categorical integer values

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

    # Transform Categorical Varables into their One-hot encoding
    features = pd.get_dummies(features, columns=categorical_columns, prefix_sep='=')
    continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

    # Add missing columns to the test dataset
    # After applying pd.get_dummies(), train_features returned 104 columns 
    # while test_features returned 103 columns
    if (train == False):
        features.insert(
            # loc = features.columns.get_loc('native_country=Holand-Netherlands'),
            loc = 75,
            column = 'native_country=Holand-Netherlands', value=0
        )
    
    # This essentially stores the column name of the categorical variables and the list of indexes of their one-hot representations, 
    # e.g. "wordclass" = [6,7,8,9,10,11,12]
    # Notice the 'sex' column, which is also our sensitive variable, has only 2 unique values
    one_hot_columns = {}
    for column_name in categorical_columns:
        ids = [i for i, col in enumerate(features.columns) if col.startswith("{}=".format(column_name))]
        if (len(ids) > 0):
            assert len(ids) == ids[-1] - ids[0] + 1
        one_hot_columns[column_name] = ids

    # Save the column name and column indexes of the final dataframe
    # column_ids = {col: idx for idx, col in enumerate(features.columns)}
        
    # Convert data to torch tensor
    x = torch.tensor( features.values.astype(np.float32), device=device )
    y = torch.tensor( labels.values.astype(np.int64), device=device )
    a = torch.tensor( a.astype(np.bool), device=device )

    # Normalize the values of the continous columns
    if (normalize):
        columns = continuous_columns if continuous_columns is not None else np.arange(x.shape[1])
        mean, std = x.mean(dim=0)[columns], x.std(dim=0)[columns]
        x[:, columns] = (x[:, columns] - mean) / std    

    return x, y, a

def load_dataset(download=True, train=True, device="cpu"):
    """
    Downloads the raw dataset files and return tensors
    """

    device = device
    if (device == None):
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    if (download):
        _adult_download()
    if (train):
        features, label, sensitive = _read_data_file( adult_download_folder + adult_filenames['train'], device=device, train=True)
    else:
        features, label, sensitive = _read_data_file( adult_download_folder + adult_filenames['test'], device=device, train=False)

    return features, label, sensitive