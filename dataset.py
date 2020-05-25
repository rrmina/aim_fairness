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

from sklearn.model_selection import train_test_split

class Adult(object):
    # Column names
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]

    # Income Label Mappings
    train_labels_map = {'<=50K': 0, '>50K': 1}
    test_labels_map = {'<=50K.': 0, '>50K.': 1}

    def __init__(self, sensitive_attribute="sex", device=None, normalize=True):
        super(Adult, self).__init__()

        # Device
        self.device = device
        if (device == None):
            self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        # Emphasis on the sensitive attribute
        self.sensitive_attribute = sensitive_attribute

        # Initiaize file paths
        self.data_dir = "data"
        train_data_file = path.join(self.data_dir, "adult.data")
        test_data_file = path.join(self.data_dir, "adult.test")

        # Check file it exists and Download if DNE
        self._check_and_download(train_data_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        self._check_and_download(test_data_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')

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

        # Categorize whether a Column is a (1) continuous variable or a (2) categorical variable
        # Column names are stored in their respective lists
        continuous_vars = []                    # We'll only store the continuous column names because their index will be changed once we turn
                                                # we turn the categorical variables into their one-hot encoding
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
        a_train = np.logical_not(pd.Categorical(train_features[self.sensitive_attribute]).codes)            # pd.Categorical.codes converts categorical strings to categorical integer values
        a_test = np.logical_not(pd.Categorical(test_features[self.sensitive_attribute]).codes)

        # Transform Categorical Variables into their One-hot encoding
        train_features = pd.get_dummies(train_features, columns=self.categorical_columns, prefix_sep='=')
        test_features = pd.get_dummies(test_features, columns=self.categorical_columns, prefix_sep='=')
        self.continuous_columns = [train_features.columns.get_loc(var) for var in continuous_vars]

        # Add mission columns to the test dataset
        # After applying pd.get_dummies, train_features returned 104 columns 
        # while test_features returned 103 columns
        test_features.insert(
            loc = train_features.columns.get_loc('native_country=Holand-Netherlands'),
            column='native_country=Holand-Netherlands', value=0
        )

        # This essentially stores the column name of the categorical variables a,d the list of indexes of their one-hot representations, 
        # e.g. "workclass" = [6,7,8,9,10,11,12] 
        # Notice that 'sex' column, which is also our sensitive attribute, has only 2 unique values
        self.one_hot_columns={}
        for column_name in self.categorical_columns:
            ids = [i for i, col in enumerate(train_features.columns) if col.startswith("{}=".format(column_name))]
            if (len(ids) > 0):
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        # print("categorical features: ", self.one_hot_columns.keys())

        # Save the column names and column indexes of the final dataframe
        self.column_ids = {col: idx for idx, col in enumerate(train_features.columns)}

        # Convert train data to torch tensor
        train_features = torch.tensor(train_features.values.astype(np.float32), device=self.device)
        train_labels = torch.tensor(train_labels.values.astype(np.int64), device=self.device)
        train_a = torch.tensor(a_train.astype(np.bool), device=self.device)

        # Divide train set into development/train set and validation set
        self.x_train, self.x_val, self.y_train, self.y_val, self.a_train, self.a_val = train_test_split(
            train_features, train_labels, train_a, test_size=0.1, random_state=0
        )

        # Convert test data to torch tensor
        self.x_test = torch.tensor(test_features.values.astype(np.float32), device=self.device)
        self.y_test = torch.tensor(test_labels.values.astype(np.int64), device=self.device)
        self.a_test = torch.tensor(a_test.astype(np.bool), device=self.device)

        # Normalize the values of the continuous variables
        # Normaluze instead of Min-Max [0,1] scaling
        if (normalize):
            self._normalize(self.continuous_columns)

    def load_dataset(self):
        return self.x_train, self.y_train, self.a_train, self.x_val, self.y_val, self.a_val, self.x_test, self.y_test, self.a_test

    def _check_and_download(self, filepath, url):
        if not path.exists(filepath):
            request.urlretrieve(url, filepath)

    def _normalize(self, columns):
        columns = columns if columns is not None else np.arange(self.x_train.shpe[1])

        self.mean, self.std = self.x_train.mean(dim=0)[columns], self.x_train.std(dim=0)[columns]

        self.x_train[:, columns] = (self.x_train[:, columns] - self.mean) / self.std
        self.x_val[:, columns] = (self.x_val[:, columns] - self.mean) / self.std
        self.x_test[:, columns] = (self.x_test[:, columns] - self.mean) / self.std