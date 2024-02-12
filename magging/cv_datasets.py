'''
    This scripts perfroms 5 fold cv on the datasets to find the optimal penalty term and then proceeds
    calculating the magging weights.
'''

import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from preprocessing import make_feature_preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from cvxopt import matrix, solvers
from itertools import combinations

# Load data from global parquet folder 
def load_data(outcome, source, version='train'):
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    relative_path = os.path.join('..', 'Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.abspath(os.path.join(current_directory, relative_path))
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 


# Include only admissions with recorded sex
_Xydata={
    'eicu': load_data('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid': load_data('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic': load_data('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv': load_data('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing('outcome', missing_indicator=True, categorical_indicator=False, lgbm=False)
    ).set_output(transform="pandas")

pipeline = Pipeline(steps=[
        ('preprocessing', Preprocessor),
        ('model', Lasso(max_iter = 10000))]
        )

datasets = ['mimic', 'hirid', 'eicu', 'miiv']
_results_groups = {dataset: {} for dataset in datasets} 
for r in range(2,len(datasets)):
        for group_combination in combinations(datasets,r):
             print(group_combination)

for dataset in datasets:
    print(f'Start with CV on {dataset}')
    search = GridSearchCV(pipeline, param_grid={'model__alpha': np.linspace(0.01,10,50)})
    search.fit(_Xydata[dataset], _Xydata[dataset]['outcome'])
    for r in range(2,len(datasets)):
        for group_combination in combinations(datasets,r):
            if dataset in group_combination:
                _results_groups[dataset][group_combination] = {
                    'alpha': search.best_params_['model__alpha'],
                    '_coef': search.best_estimator_.named_steps['model'].coef_,
                    'pred': np.vstack((search.predict(pd.concat([_Xydata[group] for group in group_combination]))))
                }
    print(f'Done with {dataset}')


_magging_results = {dataset: {} for dataset in datasets}
for r in range(2, len(datasets)):
    for group_combination in combinations(datasets,r):
        n_obs = pd.concat([_Xydata[group] for group in group_combination]).shape[0]
        fhat = np.column_stack([_results_groups[group][group_combination]['pred'] for group in group_combination])
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

        # Solve the quadratic program to obtain magging weights
        solution = solvers.qp(P, q, G, h, A, b)
        w = np.array(solution['x']).round(decimals=4).flatten() # Magging weights
        print(group_combination, ' with Magging weights: ', w)
        for dataset in datasets:
            if dataset not in group_combination:
                predictions = []
                for group in group_combination:
                    pipeline.named_steps['model'].alpha = _results_groups[group][group_combination]['alpha']
                    pipeline.fit(_Xydata[group], _Xydata[group]['outcome'])
                    predictions.append(np.array(pipeline.predict(_Xydata[dataset])))
                if predictions:
                    y_pred = np.dot(w, predictions)
                    _magging_results[dataset][group_combination] = {
                        'weights': w,
                        'mse Magging':  mean_squared_error(_Xydata[dataset]['outcome'], y_pred),
                        'Matrix p.d.': all(np.linalg.eigvals(H) > 0),
                        'mse single groups': [mean_squared_error(_Xydata[dataset]['outcome'], prediction) for prediction in predictions],
                        'alpha': [_results_groups[group][group_combination]['alpha'] for group in group_combination]
                    }
print(_magging_results)
print(pd.DataFrame(_magging_results))
#pd.DataFrame(_results).to_latex()
print('Script run successfull')
