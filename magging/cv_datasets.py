'''
    This script calculates the magging estimator for for combinations of different datasets and its prediction as follows: 
        1. Perfrom 5 fold cv on the individual groups to find the optimal penalty terms
        2. Calculate in-group regression coefficients (via Lasso)
        3. Calculates the magging weights
        4. Use Magging to predict on groups that have NOT been used to calculate the weights 
'''

import os
import pandas as pd
import numpy as np

from cvxopt import matrix, solvers
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

from preprocessing import make_feature_preprocessing

outcome = 'hr'
model = 'rf'

assert model in ['lasso', 'rf'], 'model not specified'

if model == 'lasso':
    regressor = Lasso(max_iter=10000)
    hyper_prameters = {'model__alpha': np.linspace(0.01,10,50)}
    preprocessor = ColumnTransformer(
        transformers=make_feature_preprocessing(missing_indicator=True, categorical_indicator=True)
        ).set_output(transform="pandas")
elif model == 'rf':
    regressor = LGBMRegressor(boosting_type='rf')
    preprocessor = ColumnTransformer(
        transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
        ).set_output(transform="pandas")
    hyper_prameters = {
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__n_estimators': [100, 800],
        'model__num_leaves': [50, 200, 1024],
        'model__feature_fraction': [0.5, 0.9],
        'model__verbose': [-1]
        }

# Load data from parquet folder in parent direction
def load_data(outcome, source, version='train'):
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    relative_path = os.path.join('..', 'Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.abspath(os.path.join(current_directory, relative_path))
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 

# Include only admissions where sex is recorded
datasets = ['eicu', 'hirid', 'mimic', 'miiv']
Xy_data={source: load_data('hr',source)[lambda x: x['sex'].isin(['Male', 'Female'])] for source in datasets}

pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', regressor)]
    )

# Estimate in-group coefficients and use them to predict on the entire dataset  
results_groups = {dataset: {} for dataset in datasets} 
for dataset in datasets:
    print(f'Start with CV on {dataset}')
    search = GridSearchCV(pipeline, param_grid=hyper_prameters)
    search.fit(Xy_data[dataset], Xy_data[dataset]['outcome'])
    for r in range(2,4):
        for group_combination in combinations(datasets, r): 
            if dataset in group_combination: 
                if model == 'lasso':
                    results_groups[dataset][group_combination] = {
                        'alpha': search.best_params_['model__alpha'],
                        '_coef': search.best_estimator_.named_steps['model'].coef_,
                        'pred': search.predict(pd.concat([Xy_data[group] for group in group_combination]))
                    }
                elif model == 'rf': 
                    results_groups[dataset][group_combination] = {
                        'hyper parameters': search.best_params_,
                        'pred': search.predict(pd.concat([Xy_data[group] for group in group_combination]))
                    }
    print(f'Done with {dataset}')

# Calculate the magging prediction
magging_results = {dataset: {} for dataset in datasets}
for r in range(2, len(datasets)):
    for group_combination in combinations(datasets,r):
        # Solve the quadratic program to obtain magging weights
        n_obs = pd.concat([Xy_data[group] for group in group_combination]).shape[0]
        fhat = np.column_stack([results_groups[group][group_combination]['pred'] for group in group_combination])
        H = fhat.T @ fhat / n_obs
        if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
            print("Attention: Matrix H is not positive definite")
            H += 1e-5
        P = matrix(H)
        q = matrix(np.zeros(r))
        G = matrix(-np.eye(r))
        h = matrix(np.zeros(r))
        A = matrix(1.0, (1, r))
        b = matrix(1.0)
        solution = solvers.qp(P, q, G, h, A, b)
        w = np.array(solution['x']).round(decimals=4).flatten() # Magging weights
        print(group_combination, ' with Magging weights: ', w)
        # Calculate the magging prediction on all groups that have NOT been used to calculate the weights
        for dataset in datasets:
            if dataset not in group_combination:
                predictions = []
                for group in group_combination:
                    if model == 'lasso':
                        pipeline.alpha = results_groups[group][group_combination]['alpha']
                    elif model == 'rf': 
                        for key, value in results_groups[group][group_combination]['hyper parameters'].items():
                            pipeline.set_params(**{key: value})
                    pipeline.fit(Xy_data[group], Xy_data[group]['outcome'])
                    predictions.append(np.array(pipeline.predict(Xy_data[dataset])))
                if predictions: 
                    y_pred = np.dot(w, predictions)
                    if model == 'lasso': 
                        magging_results[dataset][group_combination] = {
                            'weights': w,
                            'mse Magging':  mean_squared_error(Xy_data[dataset]['outcome'], y_pred),
                            'Matrix p.d.': all(np.linalg.eigvals(H) > 0),
                            'mse single groups': [mean_squared_error(Xy_data[dataset]['outcome'], prediction) for prediction in predictions],
                            'alpha': [results_groups[group][group_combination]['alpha'] for group in group_combination]
                        }
                    elif model == 'rf':
                        y_pred = np.dot(w, predictions)
                        magging_results[dataset][group_combination] = {
                            'weights': w,
                            'mse Magging':  mean_squared_error(Xy_data[dataset]['outcome'], y_pred),
                            'Matrix p.d.': all(np.linalg.eigvals(H) > 0),
                            'mse single groups': [mean_squared_error(Xy_data[dataset]['outcome'], prediction) for prediction in predictions],
                            'alpha': [results_groups[group][group_combination]['hyper parameters'] for group in group_combination]
                        }
print('Magging Results: ', magging_results)
print(pd.DataFrame(magging_results))
pd.DataFrame(magging_results).to_parquet(f'parquet/{outcome}/magging_results.parquet')
pd.DataFrame(magging_results).to_latex()
print('Script run successfull')