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

### to do: 
#### - compare predictive performance
#### - generalize it 
#### - new groups
##### - region (exists only for eICU)
##### - hospital_id
##### - services (exclude all with less than 3 admissions --- all other have at least 8 observations)
###### - region and hospital_id, services have different size of feature vector for eicu
###### - mimic,miiv,hirid has none values (wenn in DataFrame eingefügt und länge=0 dann skippen)
###### Lösung: Merke welche column 0 wird und dort eine 0 hinzufügen
#### - non-linear estimators (plug-in principle)
##### - LGBM
##### - Anchor + LGBM
#### - Plotting
#### - Make GitHub look nice
#### - Shared Lasso
#### - Group DRO
#### - Write Overleaf document for Malte
# Ideen
## - Schau bei einzelnen Gruppen nach ob outlier oder ähnliches vorliegen
## - Schau histogram/Box plots der einzelnen Gruppen an ob da ein shift zu erkennen ist --> ggf. Methode zum estimaten innerhalb der Gruppen anpassen

Regressor='elasticnet'
datasets=['eicu','hirid','mimic','miiv']
grouping_column = 'region'


########################### Check if region, hospital_id, services have now same size

'''_data= load_data(f'parameters_{Regressor}_{grouping_column}')['eicu'].values
print(_data)
for i in range(len(_data)):
    print(len((_data[i])['coefficients']))
    print()
raise ValueError'''

hyper_params={
    'elasticnet':{"alpha": [0.001, 1, 10, 100],
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

def load_data_parquet(outcome, source, version='train'):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.join(current_directory, relative_path)
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 

_Xydata={
    'eicu':load_data_parquet('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid':load_data_parquet('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic':load_data_parquet('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv':load_data_parquet('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}

def fit_and_extract_info(group_df, grouping_column, dataset):
    if len(group_df)<8:
        return {}
    search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_params[Regressor].items()})
    search.fit(group_df, group_df['outcome'])
    # Extract alpha and l1_ratio from the best_params dictionary
    best_params = search.best_params_
    alpha = best_params['model__alpha']
    l1_ratio = best_params['model__l1_ratio']
    print(search.best_estimator_.named_steps['model'].feature_names_in_)
# Extract coefficients and intercept from the fitted ElasticNet model
    coefficients = search.best_estimator_.named_steps['model'].coef_
    intercept = search.best_estimator_.named_steps['model'].intercept_
    return {
        f'{grouping_column}': group_df[grouping_column].iloc[0],  # Extract the group
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'coefficients': coefficients,
        'intercept': intercept
    }

#print('Start grouped regression')


if not results_exist(f'parameters_{Regressor}_{grouping_column}_data.parquet'): # include not
    grouped_dfs = {}
    _params = {'eicu':{},
                'hirid':{},
                'mimic':{},
                'miiv':{}
    }
    for dataset in datasets:
        print(dataset)
        grouped=_Xydata[dataset].groupby(by=grouping_column)
        for group_name, group_data in grouped:
            _params[dataset][group_name]=fit_and_extract_info(group_data,grouping_column,dataset)
    raise ValueError
    save_data(path=f'parameters_{Regressor}_{grouping_column}_data.parquet',results=_params)


'''
if results_exist(f'parameters_{Regressor}_{grouping_column}_data.parquet'): # include not
    print(f'Start {grouping_column}')
    grouped_dfs = {}
    _params = {'eicu':{},
                'hirid':{},
                'mimic':{},
                'miiv':{}
    }

    for dataset in datasets:
        print(dataset)
        print(len(_Xydata[dataset].columns))
        grouped=_Xydata[dataset].groupby(by=grouping_column)
        missing_columns = []

        for group_name, group_data in grouped:
            missing_cols = group_data.columns[group_data.isna().all()]
            missing_columns.extend(missing_cols)
        # Remove duplicates by converting to a set and back to a list
        missing_columns = list(set(missing_columns))
        for group_name, group_data in grouped:
            group_data.drop(columns=missing_columns, inplace=True)
        _Xydata[dataset].drop(columns=missing_columns,inplace=True)
        print(len(_Xydata[dataset].columns))
        grouped_dfs[dataset] = pd.DataFrame(list(_Xydata[dataset].groupby(by=grouping_column).apply(fit_and_extract_info, grouping_column, dataset)))
        # Filter out None values (groups with less than min_observation_count observations)
        grouped_dfs[dataset] = grouped_dfs[dataset].dropna()
        
        # Continue with your existing code to populate _params
        for group_id, row in grouped_dfs[dataset].iterrows():
            _params[dataset][row[grouping_column]] = {
                'parameters': row['coefficients'],
                'intercept': row['intercept'],
                'alpha': row['alpha'],
                'l1_ratio': row['l1_ratio']
            }
    raise ValueError
    save_data(path=f'parameters_{Regressor}_{grouping_column}_data.parquet',results=_params)
    print(f'finished {grouping_column}')
    print()'''


def preprocessing(outcome, dataset_name): # drop all rows with missing sex and replace bool dtype as float dtype since numpy can't do calculus with bool values
    Xy_data=load_data_parquet(outcome,dataset_name)
    X=Preprocessor.fit_transform(Xy_data)
    s=X.select_dtypes(include='bool').columns
    X[s]=X[s].astype('float')
    if dataset_name=='eicu':
        X=X[X['categorical__sex_None']==0].drop(columns=['categorical__sex_None'])
    return X

_Xdata={
    'eicu':preprocessing('hr','eicu'),
    'hirid':preprocessing('hr','hirid'),
    'mimic':preprocessing('hr','mimic'),
    'miiv':preprocessing('hr','miiv')
}


### Spezialisiert auf Predictions von eICU auf alles andere
_params_model=load_data(f'parameters_{Regressor}_{grouping_column}')['eicu'].dropna().values
error=[] 
for target in datasets:
    r=len(_params_model)
    theta = np.column_stack([(_params_model[i])['coefficients'] for i in range(len(_params_model))])
    hatS = _Xdata[target].T @ _Xdata[target] / _Xdata[target].shape[0]
    H = (theta.T @ hatS @ theta).values # needs to be positive definite
    P = matrix(2 * H)
    q = matrix(np.zeros(r))
    G = matrix(-np.eye(r))
    h = matrix(np.zeros(r))
    A = matrix(1.0, (1, r))
    b = matrix(1.0)
    solution = solvers.qp(P, q, G, h, A, b)
    w = np.array(solution['x']).round(decimals=2).flatten()
    
    # Set model parameters, intercept and penalization
    model_coef = np.sum([w[i] * (_params_model[i])['coefficients'] for i in range(len(_params_model))], axis=0)
    model_penal = np.sum([w[i] * (_params_model[i])['alpha'] for i in range(len(_params_model))], axis=0)
    model_l1_ratio = np.sum([w[i] * (_params_model[i])['l1_ratio'] for i in range(len(_params_model))], axis=0)
    model_intercept = np.sum([w[i] * (_params_model[i])['intercept'] for i in range(len(_params_model))])
    pipeline.named_steps['model'].set_params(alpha=model_penal,l1_ratio=model_l1_ratio)
    pipeline.named_steps['model'].coef_ = model_coef
    pipeline.named_steps['model'].intercept_ = model_intercept
    
    # Make predictions and calculate MSE
    y_pred = pipeline.predict(_Xydata[target])
    mse = mean_squared_error(_Xydata[target]['outcome'], y_pred)
    
    # Store results in the error list
    error.append({
        'target': target,
        'mse': round(mse,2),
        'mean alpha': round(model_penal,2),
        'mean l1_ratio': round(model_l1_ratio,2),
        'weights': w,
    })
print(pd.DataFrame(error))
print(pd.DataFrame(error)[['weights']].to_string())
print(pd.DataFrame(error).to_latex())
save_data(path=f'magging_{Regressor}_{grouping_column}_results.parquet',results=error)



"""
for group_by_column in grouping_columns:
    _params_model=load_data(f'parameters_{Regressor}_{group_by_column}') 
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
    for target in datasets:
        data = _Xydata[target]
        for r in range(1, len(datasets)):
            for source_combination in combinations(datasets, r):
                if target not in source_combination:
                    ###### Wie sehen die parameter aus wenn es "keine" Gruppe gab aka der gesamte Datensatz eine Gruppe ist
                    ###### Wie kann ich das beim Plotten hinzufügen??????
                    ###### Beschränke dich zunächst beim Plotten nur auf die Transferabiltät von einem Datensatz auf einen anderen!
                    theta = np.column_stack([[_params[source]][group_by_column]['parameters'] for source in source_combination for group_id in _params[source]])
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
                    model_coef = np.sum([w[i] * _params[source]['parameters'] for i, source in enumerate(source_combination)], axis=0)
                    model_penal = np.sum([w[i] * _params[source]['alpha'] for i, source in enumerate(source_combination)], axis=0)
                    model_l1_ratio = np.sum([w[i] * _params[source]['l1_ratio'] for i, source in enumerate(source_combination)], axis=0)
                    model_intercept = np.sum([w[i] * _params[source]['intercept'] for i, source in enumerate(source_combination)])
                    pipeline.named_steps['model'].set_params(alpha=model_penal,l1_ratio=model_l1_ratio)
                    pipeline.named_steps['model'].coef_ = model_coef
                    pipeline.named_steps['model'].intercept_ = model_intercept
                    
                    # Make predictions and calculate MSE
                    y_pred = pipeline.predict(data)
                    mse = mean_squared_error(data['outcome'], y_pred)
                    
                    # Store results in the error list
                    error.append({
                        'target': target,
                        'group 1': source_combination[0] if len(source_combination) > 0 else '',
                        'group 2': source_combination[1] if len(source_combination) > 1 else '',
                        'group 3': source_combination[2] if len(source_combination) > 2 else '',
                        'mean alpha': model_penal,
                        'mean l1_ratio': model_l1_ratio,
                        'weights': w,
                        'mse': mse
                    })
    save_data(path=f'magging_{Regressor}_{group_by_column}_results.parquet',results=error)"""




################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
'''# Find penalization parameter on all datasets
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
                w = np.array(solution['x']).round(decimals=2).flatten()
                
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
                    'mse': round(mse,2),
                    'mean alpha': round(model_penal,2),
                    'mean l1_ratio': round(model_l1_ratio,2),
                    'weights': w
                })
save_data(path=f'magging_{Regressor}_results.parquet',results=error)
data=load_data_plotting(path=f"Parquet/magging_{Regressor}_results.parquet")
print(data.to_latex())
print("Script run successful")'''