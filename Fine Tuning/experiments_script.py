'''
    Loads the data and performs either grid search (on train) or fine-tuning (on test) evaluation 
    on the model specified in set_up.py
    Fine-tuning approach: Instead of selecting hyperparameters through the usual method of minimizing 
    the MSE via cv on the training data, the approach evaluates the MSE via CV on a "fine-tuning" dataset
    (consisting of n fine-tuning datapoints) derived from the test data. 
    This fine-tuning dataset is excluded from the final evaluation but is used specifically for 
    tuning hyperparameters. To achieve high generalization, we consider various sample seeds. 
    The optimal tuning hyperparameters are selected during the process of plotting().
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import itertools
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from set_up import hyper_parameters, model, outcome, n_seeds, sources, training_source, regressor, preprocessing, n_fine_tuning
from icu_experiments.preprocessing import make_feature_preprocessing
from icu_experiments.load_data import load_data_for_prediction
from methods import RefitLGBMRegressor
from plotting import plotting

# load the data
_data = load_data_for_prediction(outcome=outcome)

# Sex is a categorical variable
if model == 'lgbm' or model == 'rf' or model == 'lgbm_refit':
    sex_index = _data[training_source]['train'].columns.get_loc('sex')
    _data[training_source]['train']['sex'] = _data[training_source]['train']['sex'].astype('category')

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', regressor)
])

boosting_methods=['anchor_lgbm']
refit_methods=['lgbm_refit']

# Classical approach: Perform grid search on train data and evaluation for the specified model using the 
# optimal hyperparameter found on the train data
if not os.path.exists(f'parquet/{model}_grid_results.parquet') and model not in boosting_methods: 
    mse_grid_search = []
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
                mse_grid_search.append({
                    'parameters': search.best_params_,
                    'target': source,
                    'MSE': mse
                    })
            else: 
                mse_grid_search.append({
                    'parameters': None,
                    'target': source,
                    'MSE': mse
                    })
        print(f'Completed {model} run on {source}')
    data = pd.DataFrame(mse_grid_search)
    data.to_parquet(f'parquet/{model}_grid_results.parquet')

# Fine-tuning approach: Use n fine-tuning datapoints from the test data and evaluate hyperparameter based
# on these fine-tuning datapoints via the MSE. The optimal hyperparameter wil be chosen by 
# in plotting()
if not os.path.exists(f'parquet/{model}_results.parquet'): 
    sample_seeds = list(range(n_seeds))
    results = []
    if model in ['ridge','lgbm','rf','anchor']:
        print(f"Hyperparametrs for {model} will be chosen via performance on fine-tuning set from target \n")
        hyper_para_combinations = list(itertools.product(*hyper_parameters[model].values()))
        num_combinations = len(hyper_para_combinations)
        for comb, hyper_para_set in enumerate(itertools.product(*hyper_parameters[model].values())):
            hyper_para = dict(zip(hyper_parameters[model].keys(), hyper_para_set))            
            pipeline.named_steps['model'].set_params(**hyper_para)
            pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
            for source in sources: 
                if source != training_source:
                    # Use different sample seeds for the fine-tuning set 
                    for sample_seed in sample_seeds:      
                        Xy_target_tuning = _data[source]["test"].sample(
                        frac=1, random_state=sample_seed # control shuffeling
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
            print(f'finished combination {comb+1} from {num_combinations} using {model}')
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
    elif model in boosting_methods:
        parts=model.split('_')
        base=parts[0]
        booster=parts[1]
        hyperparameter_combinations_base = list(itertools.product(*hyper_parameters[model][base].values()))
        hyperparameter_combinations_booster = list(itertools.product(*hyper_parameters[model][booster].values()))
        num_combinations = len(hyperparameter_combinations_base)*len(hyperparameter_combinations_booster)
        for comb_base, hyperparameter_set_base in enumerate(itertools.product(*hyper_parameters[model][base].values())):
            hyperparameter_base = dict(zip(hyper_parameters[model][base].keys(), hyperparameter_set_base))
            for comb_booster, hyperparameter_set_booster in enumerate(itertools.product(*hyper_parameters[model][booster].values())):
                hyperparameter_booster = dict(zip(hyper_parameters[model][booster].keys(), hyperparameter_set_booster))     
                pipeline.named_steps['model'].set_params(anchor_params= hyperparameter_base, lgbm_params=hyperparameter_booster)
                pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
                for source in sources: 
                    if source != training_source:
                        for sample_seed in sample_seeds: 
                            Xy_target_tuning = _data[source]["test"].sample(
                            frac=1, random_state=sample_seed
                            )
                            if Xy_target_tuning.dropna(axis=1).shape[1] == Xy_target_tuning.shape[1]:
                                Xy_target_evaluation = _data[source]['train']
                                for n in n_fine_tuning:
                                    y_pred_tuning = pipeline.predict(Xy_target_tuning[:n])
                                    y_pred_evaluation = pipeline.predict(Xy_target_evaluation)
                                    mse_tuning = mean_squared_error(Xy_target_tuning['outcome'][:n], y_pred_tuning)
                                    mse_evaluation = mean_squared_error(Xy_target_evaluation['outcome'], y_pred_evaluation)
                                    results.append({
                                            'comb_nr': comb_base+comb_booster,
                                            'parameters anchor': hyperparameter_base,
                                            'parameters lgbm': hyperparameter_booster, 
                                            "target": source,
                                            "n_test": n,
                                            "sample_seed": sample_seed,
                                            'mse tuning': mse_tuning,
                                            'mse target': mse_evaluation
                                        })
                print(f'finished combination {comb_base+comb_booster+1} from {num_combinations} using {model}')
    elif model in refit_methods: 
        print(f"Hyperparametrs for {model} will be chosen via performance on fine-tuning set from target \n")
        hyper_para_combinations = list(itertools.product(*hyper_parameters['lgbm'].values()))
        num_combinations = len(hyper_para_combinations)
        for comb, hyper_para_set in enumerate(itertools.product(*hyper_parameters['lgbm'].values())):
            hyper_para = dict(zip(hyper_parameters['lgbm'].keys(), hyper_para_set))
            pipeline_lgbm = Pipeline(steps=[
                ('preprocessing', preprocessing),
                ('model', LGBMRegressor())
            ])
            pipeline_lgbm.named_steps['model'].set_params(**hyper_para)
            pipeline_lgbm.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
            hyper_parameters_refit={
                "decay_rate": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            }
            preprocessing_lgbm=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
                                    ).set_output(transform="pandas")
            pipeline = Pipeline(steps=[
                ('preprocessing', preprocessing_lgbm),
                ('model', RefitLGBMRegressor())
            ])
            for hyper_para_set_refit in itertools.product(*hyper_parameters_refit.values()):
                hyper_para_refit = dict(zip(hyper_parameters_refit.keys(), hyper_para_set_refit))
                pipeline.named_steps['model'].set_params(prior=pipeline_lgbm.named_steps['model'], **hyper_para_refit)
                for source in sources: 
                    if source != training_source:
                        for sample_seed in sample_seeds:  
                            Xy_target_tuning = _data[source]["test"].sample(
                            frac=1, random_state=sample_seed
                            )
                            Xy_target_evaluation = _data[source]['train']
                            for n in n_fine_tuning:
                                pipeline.fit(Xy_target_tuning[:n],Xy_target_tuning['outcome'][:n])
                                y_pred_evaluation = pipeline.predict(Xy_target_evaluation)
                                mse_evaluation = mean_squared_error(Xy_target_evaluation['outcome'], y_pred_evaluation)
                                results.append({
                                        'comb_nr': comb,
                                        'parameters': hyper_para,
                                        'decay_rate': hyper_para_refit['decay_rate'],
                                        "target": source,
                                        "n_test": n,
                                        "sample_seed": sample_seed,
                                        'mse target': mse_evaluation
                                    })
            print(f'finished combination {comb+1} from {num_combinations} using {model}')
    data = pd.DataFrame(results)
    data.to_parquet(f'parquet/{model}_results.parquet')
plotting(sources=sources, training_source=training_source)

print('Script completed with no erros\n')