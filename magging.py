from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.linalg import eigvals
from data import save_data, results_exist, load_data, load_data_plotting
from itertools import combinations
import pandas as pd
import os 
import pickle
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error


Regressor='ridge'

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
    'ridge':{'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]},
    'lasso':{'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
}


Model={
    'ridge':Ridge()
}


Preprocessor=ColumnTransformer(transformers=
                                make_feature_preprocessing(missing_indicator=True)
                                ).set_output(transform="pandas")


pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessor),
    ('model', Model[Regressor])
])


datasets=['eicu','hirid','mimic','miiv']


"""
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
            f'Parameters on {source}': model_params,
            'intercept': best_model.named_steps['model'].intercept_
        })
        print(f'Completed on {source}')

    save_data(path='parameters_ridge_data.pkl',results=params)

_hyper=load_data_plotting(path='Pickle/parameters_ridge_data.pkl')

_df_hyper=pd.DataFrame(_hyper)
Xy_eicu=load_data_parquet('hr', 'eicu')
print(Xy_eicu['region'].unique())
#raise ValueError
Xy_hirid=load_data_parquet('hr', 'hirid')
Xy_mimic=load_data_parquet('hr', 'mimic')
Xy_miiv=load_data_parquet('hr', 'miiv')

X_eicu=Preprocessing.fit_transform(Xy_eicu)
X_hirid=Preprocessing.fit_transform(Xy_hirid)
X_mimic=Preprocessing.fit_transform(Xy_mimic)
X_miiv=Preprocessing.fit_transform(Xy_miiv)

s=X_eicu.select_dtypes(include='bool').columns
X_eicu[s]=X_eicu[s].astype('float')
X_hirid[s]=X_hirid[s].astype('float')
X_mimic[s]=X_mimic[s].astype('float')
X_miiv[s]=X_miiv[s].astype('float')

column_index = X_eicu.columns.get_indexer(['categorical__sex_None'])[0]

X_eicu=X_eicu[X_eicu['categorical__sex_None']==0].drop(columns=['categorical__sex_None'])

_Xdata={
    'eicu':X_eicu,
    'hirid':X_hirid,
    'mimic':X_mimic,
    'miiv':X_miiv
}"""


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
            'alpha': search.best_params_,
            'intercept': best_model.named_steps['model'].intercept_,
            f'Parameters on {source}': model_params
        })
    save_data(path=f'parameters_{Regressor}_data.parquet',results=params)


################################################################################################ Set parameters for datasets and include intercept
_params_model=load_data(f'parameters_{Regressor}') 

_params={
    'eicu':{'parameters':_params_model['Parameters on eicu'][0],
            'intercept':_params_model['intercept'][0]},
    'hirid':{'parameters':_params_model['Parameters on hirid'][1],
             'intercept':_params_model['intercept'][1]},
    'mimic':{'parameters':_params_model['Parameters on mimic'][2],
             'intercept':_params_model['intercept'][2]},
    'miiv':{'parameters':_params_model['Parameters on miiv'][3],
            'intercept':_params_model['intercept'][3]}
}


error=[]

# use itertools
# group automatically
for target in datasets:
    data=_Xydata[target]

    for group1 in datasets:
        if group1!=target:
            pipeline.named_steps['model'].set_params={'alpha':1}
            pipeline.named_steps['model'].coef_=_params[group1]['parameters']
            pipeline.named_steps['model'].intercept_=_params[group1]['intercept']
            y_pred=pipeline.predict(data)
            mse=mean_squared_error(data['outcome'],y_pred)
            error.append({
                'target': target,
                'group 1': group1,
                'group 2': "", 
                'group 3': '',
                'mse': mse
            })
            for group2 in datasets:
                if group2 not in [target, group1]:
                    theta = np.column_stack((_params[group1]['parameters'],_params[group2]['parameters']))
                    hatS = _Xdata[target].T @ _Xdata[target] / _Xdata[target].shape[0]
                    H = (theta.T @ hatS @ theta).values
                    P = matrix(2*H)  
                    q = matrix(np.zeros(2))
                    G = matrix(-np.eye(2))
                    h = matrix(np.zeros(2))
                    A = matrix(1.0, (1, 2))  
                    b = matrix(1.0)   
                    solution = solvers.qp(P, q, G, h, A, b)
                    w = np.array(solution['x']).flatten()
                    pipeline.named_steps['model'].set_params={'alpha':1}
                    pipeline.named_steps['model'].coef_= w[0]*_params[group1]['parameters']+w[1]*_params[group2]['parameters']
                    pipeline.named_steps['model'].intercept_=w[0]*_params[group1]['intercept']+w[1]*_params[group2]['intercept']
                    y_pred=pipeline.predict(data)
                    mse=mean_squared_error(data['outcome'],y_pred)
                    error.append({
                        'target': target,
                        'group 1': group1,
                        'group 2': group2,
                        'group 3': '',
                        'weights': w,
                        'mse': mse
                    })
                    for group3 in datasets: 
                        if group3 not in [target, group1, group2]:
                            theta = np.column_stack((_params[group1]['parameters'],_params[group2]['parameters'],_params[group3]['parameters']))
                            hatS = _Xdata[target].T @ _Xdata[target] / _Xdata[target].shape[0]
                            H = (theta.T @ hatS @ theta).values
                            P = matrix(2*H)  
                            q = matrix(np.zeros(3))
                            G = matrix(-np.eye(3))
                            h = matrix(np.zeros(3))
                            A = matrix(1.0, (1, 3))  
                            b = matrix(1.0)   
                            solution = solvers.qp(P, q, G, h, A, b)
                            w = np.array(solution['x']).flatten()
                            pipeline.named_steps['model'].set_params={'alpha':1}
                            pipeline.named_steps['model'].coef_= w[0]*_params[group1]['parameters']+w[1]*_params[group2]['parameters']+w[2]*_params[group3]['parameters']
                            pipeline.named_steps['model'].intercept_=w[0]*_params[group1]['intercept']+w[1]*_params[group2]['intercept']+w[2]*_params[group3]['intercept']

                            y_pred=pipeline.predict(data)
                            mse=mean_squared_error(data['outcome'],y_pred)
                            error.append({
                                'target': target,
                                'group 1': group1,
                                'group 2': group2,
                                'group 3': group3,
                                'weights': w,
                                'mse': mse
                            })

######################################################################## Why tf is it not saving correctly????
save_data(path=f'magging_{Regressor}_results.parquet',results=error)

"""
def load_data_plotting(model=None,path=None):
    if path:
        file_path=path
    else: 
        current_directory = os.getcwd()
        relative_path = os.path.join('Pickle', f'{model}_results.pkl')
        file_path = os.path.join(current_directory, relative_path)
    with open(file_path, 'rb') as data:
        _data=pickle.load(data)
    return _data"""
'''
_data=load_data_plotting(path=f"magging_{Regressor}_results.parquet")
df_results=pd.DataFrame(_data)
print(df_results.iloc[:15])'''

print("Script run successful")