import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
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
preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")


for r in range(2,4):
    for group_combination in combinations(datasets, r):
        _Xytrain = pd.concat([_Xydata[source] for source in group_combination], ignore_index=True)
        _Xtrain = pd.concat([preprocessor.fit_transform(_Xydata[source]) for source in group_combination], ignore_index=True)
        _Xtrain.fillna(0, inplace=True)
        _ytrain = _Xytrain['outcome']
        intercept = _ytrain.mean()
        _ytrain = _ytrain - intercept

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
        r_g = {source: len(_Xydata[source])/len(_Xtrain) for source in group_combination}

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

        included_features = []
        features = _Xtrain_augmented.columns

        forward_coef = {'name': [], 'train mse': []}
        for dataset in datasets:
            if dataset not in group_combination: 
                forward_coef[f'test mse {dataset}'] = []

        for i in range(51):
            model = Lasso(fit_intercept=False, alpha=0.001)
            if i != 0:
                X = _Xtrain_augmented.drop(columns=forward_coef['name'])
            else: 
                X = _Xtrain_augmented
            name = []
            train_mse = []
            test_mse = {dataset:[] for dataset in datasets if dataset not in group_combination}
            for feature in X.columns:
                if 'passthrough' in feature: 
                    if i != 0:
                        selected_columns = forward_coef['name'].copy()
                        selected_columns.append(feature)
                    else:
                        selected_columns = [feature]
                    model.fit(_Xtrain_augmented[selected_columns], _ytrain)
                    #print(model.coef_)
                    name.append(feature)
                    train_mse.append(mean_squared_error(_ytrain, model.predict(_Xtrain_augmented[selected_columns])))
                    for dataset in datasets:
                        if dataset not in group_combination:
                            Xtest = preprocessor.fit_transform(_Xydata[dataset])
                            columns = Xtest.columns
                            Xtest.rename(columns={col: f'passthrough_features__{col}' for col in columns}, inplace=True)
                            test_mse[dataset].append(mean_squared_error(_Xydata[dataset]['outcome'] - intercept, model.predict(Xtest[selected_columns])))
            results_step_df = pd.DataFrame({'name': name, 'train mse': train_mse})
            for dataset in datasets:
                if dataset not in group_combination:
                    results_step_df[f'test mse {dataset}'] = test_mse[dataset]
            best_feature_name = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['name'].iloc[0]
            best_feature_train_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['train mse'].iloc[0]
            forward_coef['name'].append(best_feature_name)
            forward_coef['train mse'].append(best_feature_train_mse)
            for dataset in datasets:
                if dataset not in group_combination: 
                    best_feature_test_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])[f'test mse {dataset}'].iloc[0]
                    forward_coef[f'test mse {dataset}'].append(best_feature_test_mse)
        pd.DataFrame(forward_coef).to_parquet(f'dsl_results/alpha_small_train_on_{group_combination}_forward_selection_results.parquet')
        print(group_combination)
        print(pd.DataFrame(forward_coef))
        print()