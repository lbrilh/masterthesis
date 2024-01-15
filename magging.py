from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.linalg import eigvals
from data import save_data, results_exist, load_data, load_data_plotting
from itertools import combinations
import pandas as pd
import os 
from copy import copy
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error

# Groups: Hospital; sex?

Regressor='elasticnet'

'''_data=load_data_plotting(path=f"Parquet/magging_{Regressor}_results.parquet")
print(_data.to_latex(index=False))
raise ValueError'''

"""
##### Make it as a df

group_sizes=[30,100,-30]

##### ToDo:
# Select the last 30 unique elements from data as validation set
# Select first unique n elements from group size 
# Use these groups to predict the mse of the hold out test set
# Do the same for Region
# Non-Linear Estimators """


hyper_params={
    'elasticnet':{"alpha": [0.001, 0.1, 1, 10, 31.6, 100],
                  "l1_ratio": [0, 0.2, 0.5, 0.8, 1]},
}


Model={
    'elasticnet':ElasticNet()
}


Preprocessor=ColumnTransformer(transformers=
                                make_feature_preprocessing(missing_indicator=True)
                                ).set_output(transform="pandas")


pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessor),
    ('model', Model[Regressor])
])


datasets=['eicu','hirid','mimic','miiv']


def load_data_parquet(outcome, source, version='train'):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.join(current_directory, relative_path)
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 


def preprocessing(outcome, dataset_name): # drop all rows with missing sex and replace bool dtype as float dtype since numpy can't do calculus with bool values
    Xy_data=load_data_parquet(outcome,dataset_name)
    X=Preprocessor.fit_transform(Xy_data)
    s=X.select_dtypes(include='bool').columns
    X[s]=X[s].astype('float')
    if dataset_name=='eicu':
        X=X[X['categorical__sex_None']==0].drop(columns=['categorical__sex_None'])
    return X


_Xydata={
    'eicu':load_data_parquet('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid':load_data_parquet('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic':load_data_parquet('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv':load_data_parquet('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}


_Xdata={
    'eicu':preprocessing('hr','eicu'),
    'hirid':preprocessing('hr','hirid'),
    'mimic':preprocessing('hr','mimic'),
    'miiv':preprocessing('hr','miiv')
}

###########################################################################################
i=1

def fit_and_extract_info(group_df):
        
        search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_params[Regressor].items()})
        data=_Xydata["eicu"]
        search.fit(data, data['outcome'])
        # Extract alpha and l1_ratio from the best_params dictionary
        best_params = search.best_params_
        alpha = best_params['model__alpha']
        l1_ratio = best_params['model__l1_ratio']
    
    # Extract coefficients and intercept from the fitted ElasticNet model
        coefficients = search.best_estimator_.named_steps['model'].coef_
        intercept = search.best_estimator_.named_steps['model'].intercept_
        print(f'group {i} done')
        i+=1
        return {
            'hospital_id': group_df['hospital_id'].iloc[0],  # Extract the hospital_id
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'coefficients': coefficients,
            'intercept': intercept
        }

print('Start grouped regression')


_params = {
    'eicu': {},
    'hirid': {},
    'mimic': {},
    'miiv': {}
}

grouped_dfs = {}

for dataset in datasets:
    grouped_dfs[dataset] = pd.DataFrame(list(_Xydata[dataset].groupby('hospital_id').apply(fit_and_extract_info)))
    for hospital_id, row in grouped_dfs[dataset].iterrows():
        _params[dataset][hospital_id] = {
            'parameters': row['coefficients'],
            'intercept': row['intercept'],
            'alpha': row['alpha'],
            'l1_ratio': row['l1_ratio']
        }

save_data(path=f'parameters_{Regressor}_hospitals_data.parquet',results=grouped_dfs)
print("Script run successful")
"""
for target in datasets:
    data = _Xydata[target]
    
    # Handle combinations of groups
    for r in range(1, len(datasets)):
        for group_combination in combinations(datasets, r):
            if target not in group_combination:
                # Calculate weights for the selected groups
                

                ############################ must stack parameters from all group 
                theta = np.column_stack([[_params['eicu']][hospital_id]['parameters'] for hospital_id in _params['eicu']])

                hatS = _Xdata[target].T @ _Xdata[target] / _Xdata[target].shape[0]
                H = (theta.T @ hatS @ theta).values # needs to be positive definite
                P = matrix(2 * H)
                q = matrix(np.zeros(r))
                G = matrix(-np.eye(r))
                h = matrix(np.zeros(r))
                A = matrix(1.0, (1, r))
                b = matrix(1.0)
                solution = solvers.qp(P, q, G, h, A, b)
                w = np.array(solution['x']).flatten()
                
                # Set model parameters, intercept and penalization
                model_coef = np.sum([w[i] * _params[group]['parameters'] for i, group in enumerate(group_combination)], axis=0)
                model_penal = np.sum([w[i] * _params[group]['alpha'] for i, group in enumerate(group_combination)], axis=0)
                model_l1_ratio = np.sum([w[i] * _params[group]['l1_ratio'] for i, group in enumerate(group_combination)], axis=0)
                model_intercept = np.sum([w[i] * _params[group]['intercept'] for i, group in enumerate(group_combination)])
                pipeline.named_steps['model'].set_params(alpha=model_penal,l1_ratio=model_l1_ratio)
                pipeline.named_steps['model'].coef_ = model_coef
                pipeline.named_steps['model'].intercept_ = model_intercept
                
                # Make predictions and calculate MSE
                y_pred = pipeline.predict(data)
                mse = mean_squared_error(data['outcome'], y_pred)
                
                # Store results in the error list
                error.append({
                    'target': target,
                    'group 1': group_combination[0] if len(group_combination) > 0 else '',
                    'group 2': group_combination[1] if len(group_combination) > 1 else '',
                    'group 3': group_combination[2] if len(group_combination) > 2 else '',
                    'mean alpha': model_penal,
                    'mean l1_ratio': model_l1_ratio,
                    'weights': w,
                    'mse': mse
                })
save_data(path=f'magging_{Regressor}_results.parquet',results=error)




################################################################################################
# Find penalization parameter on all datasets
if not results_exist(f'parameters_{Regressor}_data.parquet'):
    params = []
    for source in datasets:
        search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_params[Regressor].items()})
        data=_Xydata[source]
        search.fit(data, data['outcome'])
        best_model = search.best_estimator_
        model_params = best_model.named_steps['model'].coef_
        params.append({
            'alpha': search.best_params_['model__alpha'],
            'l1_ratio': search.best_params_['model__l1_ratio'],
            'intercept': best_model.named_steps['model'].intercept_,
            f'Parameters on {source}': model_params
        })
    save_data(path=f'parameters_{Regressor}_data.parquet',results=params)

print('GridCV done')
_params_model=load_data(f'parameters_{Regressor}') 

_params={
    'eicu':{'parameters':_params_model['Parameters on eicu'][0],
            'intercept':_params_model['intercept'][0],
            'alpha':_params_model['intercept'][0],
            'l1_ratio':_params_model['l1_ratio'][0]},
    'hirid':{'parameters':_params_model['Parameters on hirid'][1],
             'intercept':_params_model['intercept'][1],
             'alpha':_params_model['intercept'][1],
             'l1_ratio':_params_model['l1_ratio'][1]},
    'mimic':{'parameters':_params_model['Parameters on mimic'][2],
             'intercept':_params_model['intercept'][2],
             'alpha':_params_model['intercept'][2],
             'l1_ratio':_params_model['l1_ratio'][2]},
    'miiv':{'parameters':_params_model['Parameters on miiv'][3],
            'intercept':_params_model['intercept'][3],
            'alpha':_params_model['intercept'][3],
            'l1_ratio':_params_model['l1_ratio'][3]}
}


error=[]
# Group automatically 
# Generalize it - should I generalize it / can I? Make seperate file to do calculations???
# Schöner machen im sinne von mehr gruppen: Kann einfach auf einzelenen datensätzen mehr Gruppen machen und dann zusammen addieren (ist das mehrfache Iteration von magging???) oder welchen Ansatz sollte ich verfolgen???
# Non-linear estimators
# make plots 
# Zusammenfassen / schreiben 

for target in datasets:
    data = _Xydata[target]
    
    # Handle combinations of groups
    for r in range(1, len(datasets)):
        for group_combination in combinations(datasets, r):
            if target not in group_combination:
                # Calculate weights for the selected groups
                theta = np.column_stack([_params[group]['parameters'] for group in group_combination])
                hatS = _Xdata[target].T @ _Xdata[target] / _Xdata[target].shape[0]
                H = (theta.T @ hatS @ theta).values # needs to be positive definite
                P = matrix(2 * H)
                q = matrix(np.zeros(r))
                G = matrix(-np.eye(r))
                h = matrix(np.zeros(r))
                A = matrix(1.0, (1, r))
                b = matrix(1.0)
                solution = solvers.qp(P, q, G, h, A, b)
                w = np.array(solution['x']).flatten()
                
                # Set model parameters, intercept and penalization
                model_coef = np.sum([w[i] * _params[group]['parameters'] for i, group in enumerate(group_combination)], axis=0)
                model_penal = np.sum([w[i] * _params[group]['alpha'] for i, group in enumerate(group_combination)], axis=0)
                model_l1_ratio = np.sum([w[i] * _params[group]['l1_ratio'] for i, group in enumerate(group_combination)], axis=0)
                model_intercept = np.sum([w[i] * _params[group]['intercept'] for i, group in enumerate(group_combination)])
                pipeline.named_steps['model'].set_params(alpha=model_penal,l1_ratio=model_l1_ratio)
                pipeline.named_steps['model'].coef_ = model_coef
                pipeline.named_steps['model'].intercept_ = model_intercept
                
                # Make predictions and calculate MSE
                y_pred = pipeline.predict(data)
                mse = mean_squared_error(data['outcome'], y_pred)
                
                # Store results in the error list
                error.append({
                    'target': target,
                    'group 1': group_combination[0] if len(group_combination) > 0 else '',
                    'group 2': group_combination[1] if len(group_combination) > 1 else '',
                    'group 3': group_combination[2] if len(group_combination) > 2 else '',
                    'mean alpha': model_penal,
                    'mean l1_ratio': model_l1_ratio,
                    'weights': w,
                    'mse': mse
                })
save_data(path=f'magging_{Regressor}_results.parquet',results=error)
data=load_data_plotting(path=f"Parquet/magging_{Regressor}_results.parquet")
print(data)
print("Script run successful")"""