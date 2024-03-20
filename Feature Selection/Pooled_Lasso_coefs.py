''' 
    Calculates and stores the coefficients for Pooled Lasso with response variable "outcome". 
    The coefficients are oreder by their absolute magintude. 
    An indicator is assigned indicating the sign of the coefficient (positive: 1, negative: 0).
    The penalty term alpha is chosen via 5-Fold cross-validation. 
    To mitigate the risk associated with selecting a potentially unfavorable random state during the initialization phase of Lasso calculation, 
    we employ the parameter n_states to manage the random states during initialization. Subsequently, we compute the average of the coefficients obtained from multiple initializations.
    Running this code requires CATEGORICAL_COLUMNS = ['sex'] in constants.py
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
y_train = Xy_train['outcome']
y_train = y_train - y_train.mean()

coefs_rs = {} # coefficients of each random state
for random_state in tqdm(range(n_states), desc='Grid Search n_states: '):    
    # Find best penalization parameter for each random state
    parameters = {'alpha': np.linspace(0.001, 10, 20)}
    search = GridSearchCV(Lasso(max_iter=100000, random_state=random_state, fit_intercept=False), parameters, n_jobs=-1)
    search.fit(X_train, y_train)
    coefs_rs[f'random_state {random_state}'] = search.best_estimator_.coef_

# Extract coefficients and plot them; average over random states
coefs = pd.DataFrame(coefs_rs)
coefs["Avg coefs"] = coefs.mean(axis=1)
coefs['abs_coefs'] = np.abs(coefs["Avg coefs"])
coefs['feature names'] = search.best_estimator_.feature_names_in_
coefs['color'] = (coefs['Avg coefs']>=0).astype('bool')
coefs.sort_values(by='abs_coefs', inplace=True, ascending=False)
coefs.to_parquet(f'parquet/{outcome}/Pooled_Lasso_coefs_{outcome}.parquet')