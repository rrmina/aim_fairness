# Adult dataset
# Preprocessing is largely based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/adult.py
# 
# Data                  : 112 Dimensions ( / {Sex, Income})
# Sensitive attribute   :   1 Dimension  (Sex)
# Ground Truth          :   1 Dimension  (Income)
import torch

import pandas as pd
import numpy as np

from urllib import request
from os import path

class AdultDataset(Object):
    # Column names
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]

    # Income Label Mappings
    train_labels_map = {'<=50K': 0, '>50K': 1}
    test_labels_map = {'<=50K.': 0, '>50K.': 1}

    def __init__(self, sensitive_attribute="sex"):
        super(AdultDataset, self).__init__()

        # Emphasis on the sensitive attribute
        self.sensitive_attribute = sensitive_attribute

        # Initiaize file paths
        self.data_dir = "data"
        train_data_file = path.join(self.data_dir, "adult.data")
        test_data_file = path.join(self.data_dir, "adult.test")

        # Check file it exists and Download if DNE
        _check_and_download(train_data_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        _check_and_download(test_data_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')

        # Read CSV Files
        train_dataset = pd.read_csv(train_data_file, sep=',', header=None, names=self.column_names)
        test_dataset = pd.read_csv(test_data_file, sep=',', header=0, names=self.column_names)         # The first line contains a random comment

        # Preprocess strings
        train_dataset = train_dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        test_dataset = test_dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Drop missing values
        train_dataset.replace(to_replace='?', value=np.nan, inplace=True)
        test_dataset.replace(to_replace='?', value=np.nan, inplace=True)
        train_dataset.dropna(axis=0, inplace=True)
        test_dataset.dropna(axis=0, inplace=True)

        # Encode Labels
        train_dataset.replace(self.train_labels_map, inplace=True)
        test_dataset.replace(self.test_labels_map, inplace=True)

        # Split Features and Labels
        train_features, train_labels = train_dataset.drop('income', axis=1), train_dataset['income']
        test_features, test_labels = test_dataset.drop('income', axis=1), test_dataset['income']

        # Categorize whether a Column is a continuous variable or a categorical variable
        continuous_vars = []
        self.categorical_columns = []
        for col in train_features.columns:
            if (train_features[col].isnull().sum() > 0):
                train_features.drop(col, axis=1, inplace=True)
            else:
                if (train_features[col].dtype == np.object):
                    self.categorical_columns += [col]
                else:
                    continuous_vars += [col]

        # Preprocessing the sensitive attribute
        self.sensitive_nunique = train_features[sensitive_attribute].nunique()
        sensitive_train = np.logical_not(pd.Categorical(train_features[self.sensitive_attribute]).codes)            # pd.Categorical.codes convers categorical strings to categorical integer values
        sensitive_test = np.logical_not(pd.Categorical(test_features[self.senstive_attribute]).codes)

        # Transform Categorical Variables into their One-hot encoding
        train_features = pd.get_dummies(train_features, columns=self.categorical_columns, prefix_sep='=')
        test_features = pd.get_dummies(test_features, columns=self.categorical_columns, prefix_sep='=')

    def _check_and_download(self, filepath, url):
        if not path.exists(filepath):
            request.urlretrieve(url, filepath)