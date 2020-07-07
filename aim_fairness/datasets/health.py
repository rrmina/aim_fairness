# Health Dataset
# Preprocessing is largely based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/health.py
# which is also based on https://github.com/truongkhanhduy95/Heritage-Health-Prize
# 
#   Features              : 100 Dimensions ( / {AgeAtFirstClaim, Charlson})
#   Sensitive attribute   :   1 Dimension  (AgeAtFirstClaim)
#   Ground Truth Label    :   1 Dimension  (Charlson Index)
#
#   Please note that the features does not contain the sensitive attribute. 
#   This is in contrast to the assumptions of Ruoss et. al https://github.com/eth-sri/lcifr/issues/1
#   which are by the way based on Data Producer - Data Consumer Framework of McNamara et. al
#
#   We highly suggest that you concatenate the features and sensitive instead in cases where 
#   Data Producer is assumed to have access to every feature, including the sensitive attribute

import os
import urllib.request as ur
import zipfile

import torch
import pandas as pd
import numpy as np

# External File Info
health_url = "https://foreverdata.org/1015/content/"
health_download_folder = "data/"

# File Names
health_zip_filename = "HHP_release3.zip"
health_filenames = {
    "claims": "Claims.csv",
    "drugs": "DrugCount.csv",
    "labs": "LabCount.csv",
    "members": "Members.csv"
}
health_processed_filename = "health_clean.csv"

# Column Names
column_names = ['MemberID', 'ProviderID', 'Sex', 'AgeAtFirstClaim']
claims_cat_names = ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc']

# Label
label_attribute = "max_CharlsonIndex"

# Sensitive Attribute
sensitive_attribute = "AgeAtFirstClaim"

def _download_one( filename ):
    """
    Download a file if not present
    Default save path is "data/" folder
    """

    filepath = health_download_folder + filename

    if not os.path.exists( health_download_folder ):
        os.makedirs( health_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( health_url + filename, filepath )
    else: 
        print( "Found and verified ", filepath )

def _health_download():
    _download_one( health_zip_filename )

def _read_data_file(zip_path, device, transfer, normalize=True):
    # Load Zip File
    zf = zipfile.ZipFile(zip_path)

    # Preprocess and Save or Load data frame
    if not os.path.exists( health_download_folder + health_processed_filename ):
        # Load the data frames from the Zip File
        df_claims   =   _preprocess_claims( pd.read_csv( zf.open( health_filenames[ "claims" ] )))
        df_drugs    =   _preprocess_drugs( pd.read_csv( zf.open( health_filenames[ "drugs" ] )))
        df_labs     =   _preprocess_labs( pd.read_csv( zf.open( health_filenames[ "labs" ] )))
        df_members  =   _preprocess_members( pd.read_csv( zf.open( health_filenames[ "members" ] )))

        # Merge all the data frames
        df_labs_drugs = pd.merge(df_labs, df_drugs, on=['MemberID', 'Year'], how='outer')
        df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims, on=['MemberID', 'Year'], how='outer')
        df_health = pd.merge(df_labs_drugs_claims, df_members, on=['MemberID'], how='outer')

        # Drop Unnecessary columns
        df_health.drop(['Year', 'MemberID'], axis=1, inplace=True)
        df_health.fillna(0, inplace=True)

        # Save Preprocessed Data Frame
        df_health.to_csv( health_download_folder + health_processed_filename, index=False )
    else:
        # Load Preprocessed Data Frame
        df_health = pd.read_csv( health_download_folder + health_processed_filename, sep=',')

    # In case of transfer learning task
    if (transfer):
        drop_cols = [col for col in df_health.columns if col.startswith('PrimaryConditionGroup=')]
        df_health.drop(drop_cols, axis=1, inplace=True)

    # Divide Dataset
    # Label
    if (label_attribute == "max_CharlsonIndex"):
        labels = 1 - df_health[label_attribute] 
    else:
        labels = df_health[label_attribute]

    # Sensitive Attribute
    a = np.logical_or(
        df_health['AgeAtFirstClaim=60-69'],
        np.logical_or(df_health['AgeAtFirstClaim=70-79'], df_health['AgeAtFirstClaim=80+'])
    )

    # Features
    # Drop Sensitive and Label
    senstive_drop_cols = [col for col in df_health.columns if col.startswith('AgeAtFirstClaim')]
    features = df_health.drop([label_attribute] + senstive_drop_cols, axis=1)

    # Get the location of the continuous columns
    # This will be used in normalizing their values
    continuous_vars = [col for col in features.columns if '=' not in col]
    continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

    # one_hot_columns = {}
    # for column_name in column_names:
    #     ids = [i for i, col in enumerate(features.columns) if col.startswith('{}='.format(column_name))]
    #     if len(ids) > 0:
    #         assert len(ids) == ids[-1] - ids[0] + 1
    #     one_hot_columns[column_name] = ids
    # print('categorical features: ', one_hot_columns.keys())

    # Convert data to torch tensor
    x = torch.tensor( features.values.astype(np.float32), device=device)
    y = torch.tensor( labels.values.astype(np.int64), device=device )
    a = torch.tensor( a.values.astype(np.bool), device=device) * 1

    # Normalize the values of continous columns
    if (normalize):
        columns = continuous_columns if continuous_columns is not None else np.arange(x.shape[1])
        mean, std = x.mean(dim=0)[columns], x.std(dim=0)[columns]
        x[:, columns] = (x[:, columns] - mean) / std

    return x, y, a

def load_dataset(download=True, device="cpu", transfer=False):
    device = device
    if (device == None):
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    if (download):
        _health_download()

    features, label, sensitive = _read_data_file( health_download_folder + health_zip_filename, transfer=transfer, device=device)

    return features, label, sensitive

def _preprocess_claims(df_claims):
    df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
    df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

    df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
    df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
    df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
    df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
    df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
    df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
    df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
    df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
    df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
    df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
    df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
    df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12

    df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
    df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
    df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 2
    df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 3

    df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
    df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
    df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
    df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
    df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
    df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
    df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 11
    df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
    df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
    df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 180
    df_claims['LengthOfStay'].fillna(0, inplace=True)
    df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)

    for cat_name in claims_cat_names:
        df_claims[cat_name].fillna(f'{cat_name}_?', inplace=True)
    df_claims = pd.get_dummies(df_claims, columns=claims_cat_names, prefix_sep='=')

    oh = [col for col in df_claims if '=' in col]

    agg = {
        'ProviderID': ['count', 'nunique'],
        'Vendor': 'nunique',
        'PCP': 'nunique',
        'CharlsonIndex': 'max',
        # 'PlaceSvc': 'nunique',
        # 'Specialty': 'nunique',
        # 'PrimaryConditionGroup': 'nunique',
        # 'ProcedureGroup': 'nunique',
        'PayDelay': ['sum', 'max', 'min']
    }
    for col in oh:
        agg[col] = 'sum'

    df_group = df_claims.groupby(['Year', 'MemberID'])
    df_claims = df_group.agg(agg).reset_index()
    df_claims.columns = [
                            'Year', 'MemberID', 'no_Claims', 'no_Providers', 'no_Vendors', 'no_PCPs',
                            'max_CharlsonIndex', 'PayDelay_total', 'PayDelay_max', 'PayDelay_min'
                        ] + oh

    return df_claims

def _preprocess_drugs(df_drugs):
    df_drugs.drop(columns=['DSFS'], inplace=True)
    # df_drugs['DSFS'] = df_drugs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
    df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
    df_drugs = df_drugs.groupby(['Year', 'MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
    df_drugs.columns = ['Year', 'MemberID', 'DrugCount_total', 'DrugCount_months']
    print('df_drugs.shape = ', df_drugs.shape)
    return df_drugs

def _preprocess_labs(df_labs):
    df_labs.drop(columns=['DSFS'], inplace=True)
    # df_labs['DSFS'] = df_labs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
    df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
    df_labs = df_labs.groupby(['Year', 'MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
    df_labs.columns = ['Year', 'MemberID', 'LabCount_total', 'LabCount_months']
    print('df_labs.shape = ', df_labs.shape)
    return df_labs

def _preprocess_members(df_members):
    df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
    df_members['Sex'].fillna('?', inplace=True)
    df_members = pd.get_dummies(
        df_members, columns=['AgeAtFirstClaim', 'Sex'], prefix_sep='='
    )
    print('df_members.shape = ', df_members.shape)
    return df_members
