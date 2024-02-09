
import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

sys.path.append(root_dir)

from preprocessing import make_feature_preprocessing
from constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import seaborn as sns

grouping_column = 'age_group'
age_group = True

# Load data from global parquet folder 
def load_data(outcome, source, version='train'):
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    relative_path = os.path.join('..', 'Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.abspath(os.path.join(current_directory, relative_path))
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 


# Include only admissions with recorded sex
_Xydata={
    'eicu': load_data('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid': load_data('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic': load_data('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv': load_data('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}

# Apply age grouping if needed
if age_group:
    for dataset in ['mimic', 'eicu', 'miiv', 'hirid']:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)
        _Xydata[dataset]['age_group'].dropna(inplace=True)
        print(_Xydata[dataset]['age_group'].isna().sum())
    for dataset in ['mimic', 'eicu', 'miiv', 'hirid']: 
        _Xydata[dataset]['age_group'] = _Xydata[dataset]['age_group'].cat.remove_unused_categories()

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(grouping_column, 'outcome', missing_indicator=False, categorical_indicator=True, lgbm=False)
    ).set_output(transform="pandas")


# Specify the dataset you want to create the Tukey-Anscombe plots for
dataset_to_plot = 'mimic'

Xy = Preprocessor.fit_transform(_Xydata[dataset_to_plot])

X_grouped = Xy.sort_values(by=f'grouping_column__{grouping_column}').drop(columns=['outcome__outcome'])

labels = ['child', 'young adults', 'middle age', 'senior']

diag = block_diag(X_grouped[X_grouped[f'grouping_column__{grouping_column}']==label] for label in labels)

print(diag)

X_grouped.drop(columns=[f'grouping_column__{grouping_column}'], inplace=True)