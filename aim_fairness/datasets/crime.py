# Crime Dataset
# Preprocessing is largely based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/crime.py
# 
#   Features                :  147  Dimension ( / {Crime, Race})
#   Sensitive Attribute     :    1  Dimension (ViolentCrimePerPopulation)
#   Ground Truth Label      :    1  Dimension (Race - White and (Black + Asian + Hispanic))
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
crime_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00211/'
crime_download_folder = "data/"
crime_filename = 'CommViolPredUnnormalizedData.txt'

# Column Names
column_names = [
    'communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize', 'racepctblack',
    'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
    'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
    'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
    'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
    'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
    'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar',
    'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
    'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
    'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
    'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
    'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ',
    'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters',
    'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
    'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq',
    'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
    'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
    'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
    'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders',
    'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries',
    'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
    'ViolentCrimesPerPop', 'nonViolPerPop'
]

# Sensitive Attribute
sensitive_attribute = 'race'

def _download_one( filename ):
    """
    Download a file if not present
    Default save path is "data/" folder
    """

    filepath = crime_download_folder + filename

    if not os.path.exists( crime_download_folder ):
        os.makedirs( crime_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( crime_url + filename, filepath )
    else: 
        print( "Found and verified ", filepath )

def _crime_download():
    _download_one( crime_filename )

def _read_data_file(path, device, normalize=True):
    # Load data
    data_frame = pd.read_csv(path, sep=',', header=None, names=column_names)

    # Remove features that are not predictive
    data_frame.drop(['communityname', 'countyCode', 'communityCode', 'fold'], axis=1, inplace=True)

    # Remove all other potential goal variables
    data_frame.drop(
            [
                'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults',
                'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft',
                'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop'
            ], axis=1, inplace=True
        )

    # Drop rows with missing values
    data_frame.replace(to_replace='?', value=np.nan, inplace=True)
    data_frame.dropna(axis=0, subset=['ViolentCrimesPerPop'], inplace=True)
    data_frame.dropna(axis=1, inplace=True)

    # Divide the dataset
    features, labels = data_frame.drop('ViolentCrimesPerPop', axis=1), data_frame['ViolentCrimesPerPop']

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
    
    # Preprocess Sensitive Attribute
    a = np.less(features['racePctWhite'] / 5, features['racepctblack'] + features['racePctAsian'] + features['racePctHisp'])

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
    y = torch.tensor( labels.values.astype(np.float32), device=device )
    a = torch.tensor( a.to_numpy().astype(np.bool), device=device ) * 1

    # Binarize the labels
    y = (y < y.median()) * 1

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
        _crime_download()

    features, label, sensitive = _read_data_file( crime_download_folder + crime_filename, device=device )

    return features, label, sensitive