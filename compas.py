# Compas Dataset
# Preprocessing is largely based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/compas.py
# 
#   Features              :  12 Dimensions ( / {Race, Recid})
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
compas_url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/'
compas_download_folder = 'data/'
compas_filename = 'compas-scores-two-years.csv'

# Sensitive Attribute
sensitive_attribute = 'race'

def _download_one( filename ):
    """
    Download a file if not present
    Default save path is "data/" folder
    """

    filepath = compas_download_folder + filename

    if not os.path.exists( compas_download_folder ):
        os.makedirs( compas_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( compas_url + filename, filepath )
    else: 
        print( "Found and verified ", filepath )

    return

def _compas_download():
    _download_one( compas_filename )
    return

def _read_data_file(path, device, normalize=True):
    # Load Data
    df = pd.read_csv(path)

    # Preprocess Dataset
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != '0']
    df = df[df['score_text'] != 'N/A']

    df['in_custody'] = pd.to_datetime(df['in_custody'])
    df['out_custody'] = pd.to_datetime(df['out_custody'])
    df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()

    df.drop(
        [
            'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
            'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
        ], axis=1, inplace=True
    )
    df = df[df['race'].isin(['African-American', 'Caucasian'])]

    # Split Features, Labels, and Sensitives
    features = df.drop(['is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid', sensitive_attribute], axis=1)
    features = features[[
            'age', 'sex', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
            'c_charge_desc', 'v_score_text'
        ]]
    labels = 1 - df['two_year_recid']
    sensitive = df[sensitive_attribute]

    # Preprocess Sensitive Attribute
    a = np.logical_not( pd.Categorical( sensitive ).codes )

    # Categorize whethere a column is a (1) continouus variable or a (2) categorical variable
    # We beed to store the columns names because their index will change once we convert categorical variables to one-hot encodings
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

    # Transform Categorical Variables into their One-Hot encoding
    features = pd.get_dummies(features, columns=categorical_columns, prefix_sep='=')
    continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

    # This essentially sstores the column name of the categorical variables and the list of indexes of their one-hot representations
    # e.g. "wordclass" = [6,7,8,9,10,11,12]
    # Notice the 'race' column, which is also our sensitive variable, has only 2 unique values
    one_hot_columns = {}
    for column_name in categorical_columns:
        ids = [i for i, col in enumerate(features.columns) if col.startswith("{}=".format(column_name))]
        if (len(ids) > 0):
            assert len(ids) == ids[-1] - ids[0] + 1
        one_hot_columns[column_name] = ids

    # Convert data to torch tensors
    x = torch.tensor( features.values.astype(np.float32), device=device )
    y = torch.tensor( labels.values.astype(np.int64), device=device)
    a = torch.tensor( a.astype(np.bool), device=device )

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
        _compas_download()

    features, label, sensitive = _read_data_file( compas_download_folder + compas_filename, device=device )

    return features, label, sensitive

