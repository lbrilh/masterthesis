'''
    Performs forward feature selection using Lasso regression on all combinations of two and three datasets. 
    The penalization parameter is determined through 5-fold cross-validation on the training dataset. 
    The results are saved to 'baseline_results/group_alphas_forward_selection.parquet'.
    'source' must be excluded in constants.py before running this script.
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

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

selected_alphas = {'group name': [], 'alpha': []} # optimal penalty parameter per group combination


# For each r-combination of datasets, pefrom forward selection
for r in range(2,4):
    for group_combination in combinations(datasets, r):
        
        Xytrain = pd.concat([_Xydata[source] for source in group_combination], ignore_index=True)
        Xtrain = pd.concat([preprocessor.fit_transform(_Xydata[source]) for source in group_combination], ignore_index=True)
        ytrain = Xytrain['outcome']
        intercept = ytrain.mean()
        ytrain = ytrain - intercept # All models include mean as intercept per default

        search = GridSearchCV(Lasso(fit_intercept=False, max_iter=10000000), param_grid={'alpha': np.linspace(0.001, 2, 10)})
        search.fit(Xtrain, ytrain)
        alpha = search.best_params_['alpha']
        selected_alphas['group name'].append(group_combination)
        selected_alphas['alpha'].append(alpha)

        forward_coef = {'name': [], 'train mse': []}
        for dataset in datasets:
            if dataset not in group_combination: 
                forward_coef[f'test mse {dataset}'] = []

        for i in range(Xtrain.shape[1]):
            model = Lasso(fit_intercept=False, alpha=alpha)
            if i != 0:
                X = Xtrain.drop(columns=forward_coef['name'])
            else: 
                X = Xtrain
            name = []
            train_mse = []
            test_mse = {dataset:[] for dataset in datasets if dataset not in group_combination}
            # Fit a new model for each unused feature, incorporating all previously selected features along with the current one
            for feature in X.columns:
                if i != 0:
                    selected_columns = forward_coef['name'].copy()
                    selected_columns.append(feature)
                else:
                    selected_columns = [feature]
                model.fit(Xtrain[selected_columns], ytrain)
                name.append(feature)
                train_mse.append(mean_squared_error(ytrain, model.predict(Xtrain[selected_columns])))
                for dataset in datasets:
                    if dataset not in group_combination:
                        Xtest = preprocessor.fit_transform(_Xydata[dataset])
                        test_mse[dataset].append(mean_squared_error(_Xydata[dataset]['outcome'] - intercept, model.predict(Xtest[selected_columns])))
            results_step_df = pd.DataFrame({'name': name, 'train mse': train_mse})
            for dataset in datasets:
                if dataset not in group_combination:
                    results_step_df[f'test mse {dataset}'] = test_mse[dataset]
            # Select most significant feature
            best_feature_name = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['name'].iloc[0]
            best_feature_train_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])['train mse'].iloc[0]
            forward_coef['name'].append(best_feature_name)
            forward_coef['train mse'].append(best_feature_train_mse)
            for dataset in datasets:
                if dataset not in group_combination: 
                    best_feature_test_mse = (results_step_df[results_step_df['train mse'] == results_step_df['train mse'].min()])[f'test mse {dataset}'].iloc[0]
                    forward_coef[f'test mse {dataset}'].append(best_feature_test_mse)
        # Store results for current group_combination
        pd.DataFrame(forward_coef).to_parquet(f'baseline_results/{method}/{outcome}/{outcome}_lasso_train_on_{group_combination}_forward_selection_results.parquet')


df = pd.DataFrame(selected_alphas)
df.to_parquet(f'baseline_results/{method}/{outcome}/{outcome}_group_alphas_forward_selection.parquet')