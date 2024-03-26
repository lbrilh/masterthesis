"""
    In this file, we calculate the Mean-Squared-Error of Anchor Regression as a function of gamma.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import itertools
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from ivmodels import AnchorRegression
from icu_experiments.load_data import load_data_for_prediction
from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from plotting import plot_tuning_by_gamma
from set_up import outcome, n_seeds, sources, training_source, anchor_columns, n_fine_tuning, overwrite

# Data loading function
def load_data(outcome):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet', f'{outcome}_data.parquet')
    file_path = os.path.join(current_directory, relative_path)
    if os.path.exists(file_path):
        print(f'The data file exists!')
        _data = pd.read_parquet(file_path, engine='pyarrow')
    else:
        _data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)
        data = pd.DataFrame(_data)
        data.to_parquet(f'{outcome}_data.parquet')
    print(f'Data loaded successfully: {file_path}\n')
    return _data

if overwrite: 
    # Check if data has already been processed
    _data=load_data(outcome)
    gammas = [1, 3.16, 10, 31.6, 100, 316, 1000, 3162, 10000]
    Regressor=AnchorRegression()
    Preprocessing=ColumnTransformer(transformers=
                                    make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
    pipeline = Pipeline(steps=[
        ('preprocessing', Preprocessing),
        ('model', Regressor)
        ])
    hyperparameters ={
        'instrument_regex': ['anchor'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'l1_ratio': [0, 0.2, 0.5, 0.8, 1]
        }  
    sample_seeds = list(range(n_seeds))
    results = []
    print(f"Hyperparametrs will be chosen via performance on fine-tuning set from target \n")
    hyperpara_combinations = list(itertools.product(*hyperparameters.values()))
    num_combinations = len(hyperpara_combinations)
    for gamma in gammas: 
        for comb, hyper_para_set in enumerate(itertools.product(*hyperparameters.values())):
            hyper_para = dict(zip(hyperparameters.keys(), hyper_para_set))            
            pipeline.named_steps['model'].set_params(**hyper_para, gamma=gamma)
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
                                'gamma': gamma, 
                                'comb_nr': comb,
                                'parameters': hyper_para, 
                                "target": source,
                                "n_test": n,
                                "sample_seed": sample_seed,
                                'mse tuning': mse_tuning,
                                'mse target': mse_evaluation
                            })
                        print(f'finished combination {comb+1} from {num_combinations} with sample {sample_seed} on source {source} with gamma: {gamma}')
    data = pd.DataFrame(results)
    data.to_parquet(f'parquet/anchor_results.parquet')

plot_tuning_by_gamma(sources=sources, training_source=training_source, n_tuning_points=n_fine_tuning)