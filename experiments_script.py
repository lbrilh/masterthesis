import sys
import os 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent + '/icu-experiments')
sys.path.append(parent + '/ivmodels')

from icu_experiments.load_data import load_data_for_prediction
from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, Booster
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from ivmodels import AnchorRegression
import pickle
from numpy.linalg import eig, det

print('All imports loaded successfully \n')
model = str(input('Enter model: '))
only_target = str(input('Target only? Please enter True or False. Default True: ') or 'True')
assert model in ['ols', 'rf', 'lgbm', 'anchor', 'ridge']
outcome = str(input("Enter the target variable: ") or "hr")
n_seeds = int(input('Enter number of seeds: ') or 10)

sources = ['eicu', 'hirid', 'mimic', 'miiv']

outcome_data_path = outcome + '_data.pkl'

if os.path.exists(outcome_data_path):
    print(f'The data file exists! It will be loaded from {outcome_data_path}')

    with open(outcome_data_path, 'rb') as data: 
        _data = pickle.load(data)
    
    print('Data Loading successful\n')

else:
    print(f'Start Data Loading with outcome {outcome}')
    
    _data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)

    with open(outcome_data_path, 'wb') as data:
        pickle.dump(_data, data)
        print(f'Data saved successfully to {outcome_data_path}\n')

if model in ['ols', 'ridge']:
    preprocessor = ColumnTransformer(transformers=make_feature_preprocessing(missing_indicator=True)).set_output(transform="pandas") # Allow to preprocess subbsets of data differently
    if model=='ridge':
        params = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]} 
        Regressor = Ridge() 
    elif model=='ols':
        Regressor = LinearRegression()
elif model == 'lgbm' or model == 'rf':
    sex_index = _data['eicu']['train'].columns.get_loc('sex')
    _data['eicu']['train']['sex'] = _data['eicu']['train']['sex'].astype('category')
    preprocessor = ColumnTransformer(transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)).set_output(transform="pandas") 
    Regressor = LGBMRegressor()
    if model == 'lgbm':
        params = {
            'boosting_type': ['gbdt'],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 800],
            'num_leaves': [50, 200, 1024],
            'feature_fraction': [0.5, 0.9],
            'verbose': [-1]
        }
    else: 
        params = {
            'boosting_type': ['rf'],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 800],
            'num_leaves': [50, 200, 1024],
            'feature_fraction': [0.5, 0.9],
            'verbose': [-1]
        }
elif model == 'anchor':
    anchor_columns = list(input('Please enter the anchor columns: ') or ['hospital_id'])
    preprocessor = ColumnTransformer(
        make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True) #preprocessing_steps
    ).set_output(transform="pandas")
    Regressor = AnchorRegression()
    params = {
        'gamma': [1, 3.16, 10, 31.6, 100, 316, 1000, 3162, 10000],
        'instrument_regex': ['anchor'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]
    }  


pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', Regressor)
])

mse_grid_search = {}
if only_target == 'False': 
    print("MSE")
    path_grid_results_model = model + '_grid_results.pkl'
    if os.path.exists(path_grid_results_model):
        print(f'The GridCV file for {model} exists! It will be loaded from {path_grid_results_model}')

        with open(path_grid_results_model, 'rb') as results_data: 
            model_grid_results = pickle.load(results_data)
        
        print('GridCV results loaded for {model} successful\n')
    else:
        print(f'Start with GridCV for {model}: ')
        if model not in ['ols', 'anchor']:
            search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in params.items()})
            search.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])
            print('finsihed GCV')
            pipeline.set_params(**search.best_params_)
    
        pipeline.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])

        print(f'Start evaluation with parameter selection from grid search for {model}:')
        for source in sources: 
            print(f'On {source}')
            if source != 'eicu':
                if model not in ['ols', 'anchor']:
                    mse_grid_search[model] = {'parameters': search.best_params_,
                    'MSE on {source}' : mean_squared_error(_data[source]['train']['outcome'], pipeline.predict(_data[source]['train']))}
                else: 
                    mse_grid_search[model] = {'parameters': None,
                    'MSE on {source}' : mean_squared_error(_data[source]['train']['outcome'], pipeline.predict(_data[source]['train']))}
            print(f'Completed {model} run on {source}')

        with open(path_grid_results_model, 'wb') as data:
            pickle.dump(mse_grid_search, data)
            print(f'Data saved successfully to {path_grid_results_model}')

path_results_model = model + '_results.pkl'
if os.path.exists(path_results_model):
    print(f'The results file for {model} exists! It will be loaded from {path_results_model}')

    with open(path_results_model, 'rb') as data: 
        _data = pickle.load(data)
    
    print('Evaluation data loading successful\n')
else:
    print("Begin with evaluation on tuning and test data set from target \n")

    sample_seeds = list(range(n_seeds))
    results = []

    if model not in ['ols']:
        param_combinations = list(itertools.product(*params.values()))
        num_combinations = len(param_combinations)
        for comb, param_set in enumerate(itertools.product(*params.values())):
            para = dict(zip(params.keys(), param_set))
                            
            pipeline.named_steps['model'].set_params(**para)
            pipeline.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])

            for source in sources: 
                for sample_seed in sample_seeds:      
                    if source != 'eicu':
                        Xy_target_train = _data[source]["test"].sample(
                        frac=1, random_state=sample_seed
                        )
                        Xy_target_test = _data[source]['train']
                        for n in [25, 50, 100, 200, 400, 800, 1600]:
                            y_pred_eval = pipeline.predict(Xy_target_train[:n])
                            y_pred_test = pipeline.predict(Xy_target_test)
                            
                            mse_eval = mean_squared_error(Xy_target_train['outcome'][:n], y_pred_eval)
                            mse_test = mean_squared_error(Xy_target_test['outcome'], y_pred_test)
                            
                            results.append({
                                    'comb_nr': comb,
                                    'parameters': para, 
                                    "target": source,
                                    "n_test": n,
                                    "sample_seed": sample_seed,
                                    'mse evaluation': mse_eval,
                                    'mse target': mse_test
                                })
                    print(f'finished combination {comb+1} from {num_combinations} with sample {sample_seed} on source {source} using {model}')

    elif model == 'ols': 
        results = []
        pipeline.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])
        sing = min(pipeline.named_steps['model'].singular_)
        results.append({'smalles singular value of X': sing})
        print(f'Smallest singular value of X is: {sing} \n')
        for source in sources: 
            Xy_target_test = _data[source]['train']
            y_pred_test = pipeline.predict(Xy_target_test)
            mse_test = mean_squared_error(Xy_target_test['outcome'], y_pred_test)
            results.append({
                "target": source,
                'mse target': mse_test
            })
            print(f'finished on source {source} using {model}\n')


    with open(path_results_model, 'wb') as data:
        pickle.dump(results, data)
        print(f'Data saved successfully to {path_results_model}\n')


print('Script completed with no erros\n')