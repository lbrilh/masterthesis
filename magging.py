from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.linalg import eigvals
from data import save_data, results_exist, load_data
from itertools import combinations
import pandas as pd
import os 
from cvxopt import matrix, solvers

_data=load_data('hr')

ridge_hyper={
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
}

Regressor=Ridge()
Preprocessing=ColumnTransformer(transformers=
                                make_feature_preprocessing(missing_indicator=True)
                                ).set_output(transform="pandas")

pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessing),
    ('model', Regressor)
])


if not results_exist('parameters_ridge_data.pkl'):
    params = []
    for source in ['eicu','hirid','mimic','miiv']:
        search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in ridge_hyper.items()})
        if source=='eicu':
            data=_data[source]['train']
            data=data[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
        else:
            data=_data[source]['train']
        search.fit(data, data['outcome'])
        best_model = search.best_estimator_
        model_params = best_model.named_steps['model'].coef_
        params.append({
            'alpha': search.best_params_,
            f'Parameters on {source}': model_params
        })
        print(f'Completed on {source}')

    save_data(path='parameters_ridge_data.pkl',results=params)


def load_data_parquet(outcome, source, version='train'):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.join(current_directory, relative_path)
    _data = pd.read_parquet(file_path, engine='pyarrow')
    print(f'Data loaded successfully: {file_path}\n')
    return _data 


X_eicu=Preprocessing.fit_transform(load_data_parquet('hr', 'eicu'))
X_hirid=Preprocessing.fit_transform(load_data_parquet('hr', 'hirid'))
X_mimic=Preprocessing.fit_transform(load_data_parquet('hr', 'mimic'))
X_miiv=Preprocessing.fit_transform(load_data_parquet('hr', 'miiv'))

s=X_eicu.select_dtypes(include='bool').columns
X_eicu[s]=X_eicu[s].astype('float')
X_hirid[s]=X_hirid[s].astype('float')
X_mimic[s]=X_mimic[s].astype('float')
X_miiv[s]=X_miiv[s].astype('float')

column_index = X_eicu.columns.get_indexer(['categorical__sex_None'])[0]
X_eicu=X_eicu[X_eicu['categorical__sex_None']==0]
print(X_eicu.shape)

_params_data=load_data('parameters_ridge')

params_eicu=_params_data[0]['Parameters on eicu']
params_hirid=_params_data[1]['Parameters on hirid']
params_mimic=_params_data[2]['Parameters on mimic']
params_miiv=_params_data[3]['Parameters on miiv']

mse=[]
sources=['eicu','hirid','mimic','miiv']
choose_2_combinations = list(combinations(sources, 2))

choose_3_combinations = list(combinations(sources, 3))

target='miiv'

##################### Matrix needs to be positive definite
theta = np.column_stack((params_mimic, params_hirid))
hatS = X_miiv.T @ X_miiv / X_miiv.shape[0]
H = theta.T @ hatS @ theta
P = 2 * H  
q = np.zeros(2)
G = np.eye(2)
h = np.zeros(2)
A = matrix(1.0, (1, 2))  
b = matrix(1.0)   
solution = solvers.qp(P, q, G, h, A, b)
w = np.array(solution['x']).flatten()  # Flatten to get it in the form of a 1-D array
print(w)
'''mse.append({
    'target': target,
    'used sources': {'eicu','hirid'},
    'mse': 
})'''
'''
theta = np.column_stack((params_eicu, params_mimic))
mse.append({
    'target': target,
    'used sources': {'eicu','mimic'},
    'mse': 
})

theta = np.column_stack((params_eicu, params_hirid, params_mimic))
mse.append({
    'target': target,
    'used sources': {'eicu','hirid','mimic'},
    'mse': 
})'''

'''Test when common effects are already noticable for different sizes of tuning?'''

'''_params_data=load_data('parameters_ridge.pkl')

params = pd.concat([_params_data[0]['Parameters on eicu'].
                    _params_data[1]['Parameters on hirid'],
                    _params_data[2]['Parameters on mimic'],
                    _params_data[3]['Parameters on miiv']])
'''
'''X_eicu=Preprocessing.fit_transform(_data['eicu']['train'])
X_hirid=Preprocessing.fit_transform(_data['hirid']['train'])
X_mimic=Preprocessing.fit_transform(_data['mimic']['train'])
X_miiv=Preprocessing.fit_transform(_data['miiv']['train'])
result = pd.concat([X_eicu, X_hirid, X_mimic, X_miiv], axis=0)

eigenvalues = eigvals(result)

print(eigenvalues)'''