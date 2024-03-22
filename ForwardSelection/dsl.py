'''
    This script performs forward selection and modeling using Data Shared Lasso on combinations of healthcare datasets using various regularisation strengths.
    'source' must be included in constants.py before running this script.
'''

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

outcome = 'map'
method = 'lasso'

datasets = ['eicu', 'mimic', 'miiv', 'hirid']
data = load_data_for_prediction(outcome=outcome) # load datasets with specified outcome
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in datasets}

# Initialize preprocessing of individual groups
preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

alphas = np.linspace(0.001, 2, 10) # different regularisation strengths used for DSL

# For each r-combination of datasets, pefrom forward selection
for r in range(2,4):
    for group_combination in combinations(datasets, r):
        Xy_train = pd.concat([_Xydata[source] for source in group_combination], ignore_index=True)
        X_train = pd.concat([preprocessor.fit_transform(_Xydata[source]) for source in group_combination], ignore_index=True)
        X_train.fillna(0, inplace=True)
        y_train = Xy_train['outcome']
        intercept = y_train.mean() # All models include per default mean as intercept
        y_train = y_train - intercept

        # Function to generating the augmented dataset
        interactions = ColumnTransformer(
            transformers = 
            [
                (
                    "passthrough_features",
                    "passthrough",
                    [
                        column
                        for column in X_train.columns
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
                                        for source in X_train.columns
                                        if "categorical__source" in source
                                        for column in X_train.columns
                                        if "categorical__source" not in column
                                    ]
                    ),
                    [column for column in X_train.columns],
                )
            ]
        ).set_output(transform="pandas")

        X_train_augmented = interactions.fit_transform(X_train) 

        # Control degree of sharing
        r_g = {source: len(_Xydata[source])/len(X_train) for source in group_combination}

        for column in X_train_augmented.columns: 
            if 'passthrough' not in column: 
                if 'eicu' in column: 
                    X_train_augmented[column] = 1/np.sqrt(r_g['eicu'])*X_train_augmented[column]
                elif 'mimic' in column:
                    X_train_augmented[column] = 1/np.sqrt(r_g['mimic'])*X_train_augmented[column]
                elif 'miiv' in column:
                    X_train_augmented[column] = 1/np.sqrt(r_g['miiv'])*X_train_augmented[column]
                elif 'hirid' in column: 
                    X_train_augmented[column] = 1/np.sqrt(r_g['hirid'])*X_train_augmented[column]
        
        # Forward selection for each alpha value
        forward_coef = []
        for alpha in alphas: 
            alpha_data = {'alpha': alpha, 'name': [], 'train mse': []}
            for dataset in datasets:
                if dataset not in group_combination: 
                    alpha_data[f'test mse {dataset}'] = []
            for i in range(51):
                model = Lasso(fit_intercept=False, alpha=alpha, max_iter=10000000) # Intercept has already been deducted
                X = X_train_augmented.copy()
                if i != 0:
                    X.drop(columns=alpha_data['name'], inplace=True)
                name = []
                train_mse = []
                test_mse = {dataset:[] for dataset in datasets if dataset not in group_combination}
                for feature in X.columns:
                    if 'passthrough' in feature: # Only include the shared/common features
                        if i != 0:
                            selected_columns = alpha_data['name'].copy()
                            selected_columns.append(feature)
                        else:
                            selected_columns = [feature]
                        model.fit(X_train_augmented[selected_columns], y_train)
                        name.append(feature)
                        train_mse.append(mean_squared_error(y_train, model.predict(X_train_augmented[selected_columns])))
                        # Use current model and predict on test data
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
                # Select most significant feature
                best_feature_name = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['name'].iloc[0]
                best_feature_train_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['train mse'].iloc[0]
                alpha_data['name'].append(best_feature_name)
                alpha_data['train mse'].append(best_feature_train_mse)
                for dataset in datasets:
                    if dataset not in group_combination: 
                        best_feature_test_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])[f'test mse {dataset}'].iloc[0]
                        alpha_data[f'test mse {dataset}'].append(best_feature_test_mse)
            forward_coef.append(alpha_data)
        # Save results
        pd.DataFrame(forward_coef).to_parquet(f'dsl_results/multiple alpha/{outcome}/{outcome}_multiple_alphas_train_on_{group_combination}_forward_selection_results.parquet')
