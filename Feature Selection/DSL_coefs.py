''' 
    Calculates and stores the coefficients for Data Shared Lasso with response variable "outcome". 
    The coefficients are oreder by their absolute magintude. An indicator is assigned indicating the sign of the coefficient (positive: 1, negative: 0).
    The penalty term alpha is chosen via 5-Fold cross-validation. 
    The degree of sharing is controlled using r_g. The value of r_g is the relative size of the group to the entire dataset, i.e. larger groups are forced to 
    share more than smaller groups. Each group is preprocessed individually.
    To mitigate the risk associated with selecting a potentially unfavorable random state during the initialization phase of Lasso calculation, 
    we employ the parameter n_states to manage the random states during initialization. Subsequently, we compute the average of the coefficients obtained from multiple initializations.
    Running this code requires CATEGORICAL_COLUMNS = ['sex', 'source'] in constants.py
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'map'
n_states = 10 # number of random states

datasets = ['eicu', 'mimic', 'miiv', 'hirid']
data = load_data_for_prediction(outcome=outcome)
Xy_data = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in datasets}

# Preprocessing of individual groups
preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

# create a dataset containing all data sources
Xy_train = pd.concat([Xy_data[source] for source in datasets], ignore_index=True)
X_train = pd.concat([preprocessor.fit_transform(Xy_data[source]) for source in datasets], ignore_index=True)
X_train.fillna(0, inplace=True) # fill Nan entries from categorical variable sources with 0 (i.e. not belonging to the source)
y_train = Xy_train['outcome']
y_train = y_train - y_train.mean()

# Start generating the augmented dataset
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
r_g = {source: len(Xy_data[source])/len(X_train) for source in datasets}

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

coefs_rs = {} # coefficients of each random state
for random_state in tqdm(range(n_states), desc='Grid Search n_states: '):    
    # Find best penalization parameter for each random state
    parameters = {'alpha': np.linspace(0.001, 10, 20)}
    search = GridSearchCV(Lasso(max_iter=100000, random_state=random_state, fit_intercept=False), parameters, n_jobs=-1)
    search.fit(X_train_augmented, y_train)
    coefs_rs[f'random_state {random_state}'] = search.best_estimator_.coef_

# Extract coefficients and plot them; average over random states
coefs = pd.DataFrame(coefs_rs)
coefs["Avg coefs"] = coefs.mean(axis=1)
coefs['abs_coefs'] = np.abs(coefs["Avg coefs"])
coefs['feature names'] = search.best_estimator_.feature_names_in_
coefs['color'] = (coefs['Avg coefs']>=0).astype('bool')
coefs.sort_values(by='abs_coefs', inplace=True, ascending=False)
coefs.to_parquet(f'parquet/{outcome}/dsl_coefs_{outcome}.parquet')