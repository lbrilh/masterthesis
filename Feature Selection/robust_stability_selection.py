
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'hr'
n_states = 10

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

coefs = pd.read_parquet("dsl_coefs.parquet")
intercept = (pd.read_parquet("dsl_intercepts_random_states.parquet")).mean(axis=1)

# Preprocessing of individual groups
preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

_Xytrain = pd.concat([_Xydata[source] for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
_Xtrain = pd.concat([preprocessor.fit_transform(_Xydata[source]) for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
_Xtrain.fillna(0, inplace=True)
_ytrain = _Xytrain['outcome']

# Start generating the augmented dataset
interactions = ColumnTransformer(
    transformers = 
    [
        (
            "passthrough_features",
            "passthrough",
            [
                column
                for column in _Xtrain.columns
                if "categorical__source" not in column
            ],
        ),
        (
            "interactions",
            ColumnTransformer(
                            [
                                (
                                    f"{source}_{column}".replace('categorical__source_', '').replace("__", "_"),
                                    PolynomialFeatures(
                                        degree=(2, 2),
                                        include_bias=False,
                                        interaction_only=True,
                                    ),
                                    [source, column],
                                )
                                for source in _Xtrain.columns
                                if "categorical__source" in source
                                for column in _Xtrain.columns
                                if "categorical__source" not in column
                            ]
            ),
            [column for column in _Xtrain.columns],
        )
    ]
).set_output(transform="pandas")

_Xtrain_augmented = interactions.fit_transform(_Xtrain) 

# Control degree of sharing
r_g = {source: len(_Xydata[source])/len(_Xtrain) for source in ['eicu', 'mimic', 'miiv', 'hirid']}

for column in _Xtrain_augmented.columns: 
    if 'passthrough' not in column: 
        if 'eicu' in column: 
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['eicu'])*_Xtrain_augmented[column]
        elif 'mimic' in column:
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['mimic'])*_Xtrain_augmented[column]
        elif 'miiv' in column:
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['miiv'])*_Xtrain_augmented[column]
        elif 'hirid' in column: 
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['hirid'])*_Xtrain_augmented[column]

mse = {}

mse['intercept'] = mean_squared_error(intercept, _ytrain)

num_feat = 1
for i, feature in coefs['feature names'].items():
    y_pred = intercept + _Xtrain_augmented[coefs['feature names'][:i]]*coefs['Avg coefs'][:i]
    mse[f'{num_feat} feat.'] = mean_squared_error(_ytrain, y_pred)

print(pd.DataFrame(mse))
pd.DataFrame(mse).to_parquet('robust_stability_selection.parquet')