'''
    Predict outcome using DSL on different combinations of training sets.
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
from sklearn.model_selection import GridSearchCV

from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction
from preprocessing import make_feature_preprocessing

outcome = 'hr'
method = 'lasso'

datasets = ['eicu', 'mimic', 'miiv', 'hirid']
data = load_data_for_prediction(outcome=outcome) # load datasets with specified outcome
Xy_data = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in datasets}

# Initialize preprocessing of individual groups
preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

alphas = np.linspace(0.001, 2, 10) # different regularisation strengths used for DSL

mse_results = {}
# For each r-combination of datasets, pefrom forward selection
for r in range(2,4):
    for group_combination in combinations(datasets, r):
        mse_results[group_combination] = {}
        Xy_train = pd.concat([Xy_data[source] for source in group_combination], ignore_index=True)
        X_train = pd.concat([preprocessor.fit_transform(Xy_data[source]) for source in group_combination], ignore_index=True)
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
        r_g = {source: len(Xy_data[source])/len(X_train) for source in group_combination}

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

        search = GridSearchCV(Lasso(fit_intercept=False, max_iter=1000000), param_grid={'alpha': alphas})
        search.fit(X_train_augmented, y_train)
        best_coefs = search.best_estimator_.coef_[:51]

        for target in datasets:
            if target not in group_combination:
                X_test = (preprocessor.fit_transform(Xy_data[target])).drop(columns = f'categorical__source_{target}').to_numpy()
                y_pred = intercept + X_test@best_coefs[:51]
                mse = mean_squared_error(y_pred, Xy_data[target]['outcome'])
                mse_results[group_combination][target] = mse

print(mse_results)
print(pd.DataFrame(mse_results))
print(pd.DataFrame(mse_results).to_latex())