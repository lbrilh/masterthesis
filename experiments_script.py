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
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
from numpy.linalg import eig, det
from plotting import plot_tuning
from set_up import sources, grid_search, evaluation_on_target, training_source, methods, hyper_parameters, n_fine_tuning, Regressor, Preprocessing, anchor_columns, model, outcome, n_seeds


assert model in methods


# Check if data has already been processed 
outcome_data_path = outcome + '_data.pkl'
if os.path.exists(outcome_data_path):
    print(f'The data file exists!')
    with open(outcome_data_path, 'rb') as data: 
        _data = pickle.load(data)
else:
    _data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)
    with open(outcome_data_path, 'wb') as data:
        pickle.dump(_data, data)
print(f'Data loaded successfully: {outcome_data_path}\n')


if model == 'lgbm' or model == 'rf':
    sex_index = _data[training_source]['train'].columns.get_loc('sex')
    _data[training_source]['train']['sex'] = _data[training_source]['train']['sex'].astype('category')


pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessing[model]),
    ('model', Regressor[model])
])



if grid_search == True: 
    mse_grid_search = {}
    path_grid_results_model = model + '_grid_results.pkl'
    if os.path.exists(path_grid_results_model):
        print(f'The GridCV file for {model} exists!')
        with open(path_grid_results_model, 'rb') as results_data: 
            model_grid_results = pickle.load(results_data)
    else:
        if model in ['ols']:
            print(f'No hyperparameters for {model}. Skip GridCV')
        else:
            print(f'Start with GridCV for {model}: \n')
            search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_parameters[model].items()})
            search.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
            print('finsihed GCV')
            pipeline.set_params(**search.best_params_)
        pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
        print(f'Start evaluation with parameter selection from grid search for {model}:')
        for source in sources: 
            if source != training_source:
                mse = mean_squared_error(_data[source]['train']['outcome'], pipeline.predict(_data[source]['train']))
                if model not in ['ols']:
                    mse_grid_search[model] = {
                        'parameters': search.best_params_,
                        'MSE on {source}': mse
                        }
                else: 
                    mse_grid_search[model] = {
                        'parameters': None,
                        'MSE on {source}': mse
                        }
            print(f'Completed {model} run on {source}')
        with open(path_grid_results_model, 'wb') as data:
            pickle.dump(mse_grid_search, data)
    print(f'GridCV for {model} successful\n')



if evaluation_on_target:
    path_results_model = model + '_results.pkl'
    if os.path.exists(path_results_model):
        print(f'{model} has already been evaluated!')
        with open(path_results_model, 'rb') as data: 
            _data = pickle.load(data)
    else:
        sample_seeds = list(range(n_seeds))
        results = []
        if model != 'ols':
            print(f"Hyperparametrs for {model} will be chosen via performance on fine-tuning set from target \n")
            hyper_para_combinations = list(itertools.product(*hyper_parameters[model].values()))
            num_combinations = len(hyper_para_combinations)
            for comb, hyper_para_set in enumerate(itertools.product(*hyper_parameters[model].values())):
                hyper_para = dict(zip(hyper_parameters[model].keys(), hyper_para_set))            
                pipeline.named_steps['model'].set_params(**hyper_para)
                pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
                for source in sources: 
                    if source != training_source:
                        for sample_seed in sample_seeds:      
                            Xy_target_tuning = _data[source]["test"].sample(
                            frac=1, random_state=sample_seed
                            )
                            Xy_target_evaluation = _data[source]['train']
                            for n in n_fine_tuning:
                                y_pred_tuning = pipeline.predict(Xy_target_tuning[:n])
                                y_pred_evaluation = pipeline.predict(Xy_target_evaluation)
                                mse_tuning = mean_squared_error(Xy_target_tuning['outcome'][:n], y_pred_tuning)
                                mse_evaluation = mean_squared_error(Xy_target_evaluation['outcome'], y_pred_evaluation)
                                results.append({
                                        'comb_nr': comb,
                                        'parameters': hyper_para, 
                                        "target": source,
                                        "n_test": n,
                                        "sample_seed": sample_seed,
                                        'mse tuning': mse_tuning,
                                        'mse target': mse_evaluation
                                    })
                            print(f'finished combination {comb+1} from {num_combinations} with sample {sample_seed} on source {source} using {model}')
        elif model == 'ols':
            pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
            sing = min(pipeline.named_steps['model'].singular_)
            results.append({'smalles singular value of X': sing})
            for source in sources: 
                Xy_target_evaluation = _data[source]['train']
                y_pred_evaluation = pipeline.predict(Xy_target_evaluation)
                mse_evaluation = mean_squared_error(Xy_target_evaluation['outcome'], y_pred_evaluation)
                results.append({
                    "target": source,
                    'mse target': mse_evaluation
                })
                print(f'finished on source {source} using {model}\n')
        with open(path_results_model, 'wb') as data:
            pickle.dump(results, data)
            print(f'Data saved successfully to {path_results_model}\n')

print('Script completed with no erros\n')