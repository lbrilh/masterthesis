import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction
from preprocessing import make_feature_preprocessing

outcome = 'hr'
method = 'lasso'

datasets = ['eicu', 'mimic', 'miiv', 'hirid']
data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in datasets}

# Preprocessing of individual groups
Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

for train_set in datasets: 
    _Xtrain = Preprocessor.fit_transform(_Xydata[train_set])
    _ytrain = _Xydata[train_set]['outcome']

    included_features = []
    features = _Xtrain.columns
    intercept = _ytrain.mean()
    _ytrain = _ytrain - intercept

    forward_coef = {'name': [], 'train mse': []}
    for test_set in datasets:
        if test_set != train_set:
            forward_coef[f'test mse {test_set}'] = []
    for i in range(len(_Xtrain.columns)):
        lm = LinearRegression(fit_intercept=False)
        if i != 0:
            X = _Xtrain.drop(columns=forward_coef['name'])
        else: 
            X = _Xtrain
        name = []
        train_mse = []
        test_mse = {test_set:[] for test_set in datasets if test_set != train_set}
        for feature in X.columns:
            if i != 0:
                selected_columns = forward_coef['name'].copy()
                selected_columns.append(feature)
            else:
                selected_columns = [feature]
            lm.fit(_Xtrain[selected_columns], _ytrain)
            name.append(feature)
            train_mse.append(mean_squared_error(_ytrain, lm.predict(_Xtrain[selected_columns])))
            for test_set in datasets:
                if test_set != train_set:
                    test_mse[test_set].append(mean_squared_error(_Xydata[test_set]['outcome'] - intercept, lm.predict(Preprocessor.fit_transform(_Xydata[test_set])[selected_columns])))
        results_step_df = pd.DataFrame({'name': name, 'train mse': train_mse})
        for test_set in datasets:
            if test_set != train_set:
                results_step_df[f'test mse {test_set}'] = test_mse[test_set]
        best_feature_name = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['name'].iloc[0]
        best_feature_train_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['train mse'].iloc[0]
        forward_coef['name'].append(best_feature_name)
        forward_coef['train mse'].append(best_feature_train_mse)
        for test_set in datasets:
            if test_set != train_set: 
                best_feature_test_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])[f'test mse {test_set}'].iloc[0]
                forward_coef[f'test mse {test_set}'].append(best_feature_test_mse)
    pd.DataFrame(forward_coef).to_parquet(f'baseline_results/{train_set}_forward_selection_results.parquet')
    print(train_set)
    print(pd.DataFrame(forward_coef))
    print()