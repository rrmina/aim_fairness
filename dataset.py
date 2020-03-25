# Adult dataset
# Preprocessing is largely based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/adult.py

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

    def __init__(self):
        super(AdultDataset, self).__init__()

        # Initiaize file paths
        self.data_dir = "data"
        train_data_file = path.join(self.data_dir, "adult.data")
        test_data_file = path.join(self.data_dir, "adult.test")

        # Check file it exists and Download if DNE
        _check_and_download(train_data_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        _check_and_download(test_data_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')

        # Read CSV Files
        train_dataset = pd.read_csv(train_data_file, sep=',', header=None, names=self.column_names)
        train_dataset = pd.read_csv(test_data_file, sep=',', header=0, names=self.column_names)         # The first line contains a random comment



    def _check_and_download(self, filepath, url):
        if not path.exists(filepath):
            request.urlretrieve(url, filepath)