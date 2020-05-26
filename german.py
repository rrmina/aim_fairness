# German Dataset
# Preprocessing is large based on https://github.com/eth-sri/lcifr/blob/master/code/datasets/german.py
# 
# 

import os
import urllib.request as ur

import torch
import pandas as pd
import numpy as np


# External File Info
german_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
german_download_folder = "data/"

# Column Names
column_names = [
    'status', 'months', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
    'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
    'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone',
    'foreign_worker', 'credit'
]

# Personal Status Map
personal_status_map = {'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'}

def _download_one( filename ):
    return

def _read_data_file(path, device, normalize=True)