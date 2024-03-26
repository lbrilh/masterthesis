from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from icu_experiments.constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.linalg import eigvals
import data
from data import save_data, results_exist, load_data, load_data_plotting
from itertools import combinations
import pandas as pd
import os 
from lightgbm import LGBMRegressor
from copy import copy
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error

### to do:
#### - Look at predictive performance within groups
#### - Look at Bühlmann instructions from beginning
#### - implement maybe one or two here 


#### - non-linear estimators (plug-in principle)
##### - Anchor + LGBM
#### - Make GitHub look nice
#### - Shared Lasso
#### - Group DRO
# Ideen
## - Schau bei einzelnen Gruppen nach ob outlier oder ähnliches vorliegen
## - Schau histogram/Box plots der einzelnen Gruppen an ob da ein shift zu erkennen ist --> ggf. Methode zum estimaten innerhalb der Gruppen anpassen

Regressor='lgbm'
grouping_column = 'region'
age_group = False

hyper_params={
    'elasticnet':{"alpha": [0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100],
                 "l1_ratio": [0, 0.2, 0.5, 0.8, 1]},
    'lgbm': {
        'boosting_type': ['gbdt'],
        'num_leaves': [20, 30, 40],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]}
}


Model={
    'elasticnet':ElasticNet(),
    'lgbm': LGBMRegressor(),
}


Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False, lgbm=True)
    ).set_output(transform="pandas")


pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessor),
    ('model', Model[Regressor])
])


def load_data(outcome, source, version='train'):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.join(current_directory, relative_path)
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 


#Include only admissions with recorded sex
_Xydata={
    'eicu':load_data('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid':load_data('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic':load_data('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv':load_data('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}

def find_nan_columns(grouping_column):
    """
    Input
    ----------
    grouping_column: str
        The grouping column

    Output
    ----------
    labeld_columns: dict
        The columns which consist only of missing values after grouping and added prefix numeric & missing_indicator
    """

    datasets=['eicu','hirid','mimic','miiv']

    # Include only admissions with recoreded sex
    _Xydata={
    'eicu':load_data('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid':load_data('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic':load_data('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv':load_data('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
    }

    missing_columns={} # missing value columns after grouping

    labeld_columns={} # missing value columns that include numeric and missing_indicator as prefix

    for dataset in datasets:

        missing_columns[dataset] = []

        labeld_columns[dataset]=[]

        if age_group:
            bins = [0, 15, 39, 65, float('inf')]
            labels = ['child', 'young adults', 'middle age', 'senior']

            # Use pd.cut to create a new 'age_group' column
            _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)

        grouped=_Xydata[dataset].groupby(by=grouping_column)

        for group_name, group_data in grouped:
            missing_cols = group_data.columns[group_data.isna().all()]
            missing_columns[dataset].extend(missing_cols)

        missing_columns[dataset]=list(set(missing_columns[dataset]))

        for column in missing_columns[dataset]:

            if column in NUMERICAL_COLUMNS:
                labeld_columns[dataset].append(f'numeric__{column}')
                labeld_columns[dataset].append(f'missing_indicator__missingindicator_{column}')
        
        print(f'In the dataset {dataset} the columns {labeld_columns[dataset]} will be dropped after the preprocessing')

    return labeld_columns


if not results_exist(f'parameters_{Regressor}_{grouping_column}_data.parquet'):

    """

    Calculate the feature parameters within each group. Any columns that may contain missing value columns are dropped.

    """
    # The parameter vector for each dataset
    _models = {'eicu':{},
                'hirid':{},
                'mimic':{},
                'miiv':{}
    }

    datasets=['eicu','hirid','mimic','miiv']

    for dataset in datasets:

        if age_group:
            bins = [0, 15, 39, 65, float('inf')]
            labels = ['child', 'young adults', 'middle age', 'senior']

            # Use pd.cut to create a new 'age_group' column
            _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)
            print(_Xydata[dataset]['age_group'])


        grouped=_Xydata[dataset].groupby(by=grouping_column)

        for group_name, group_data in grouped:
            
            # Skip all groups with less than 8 observations
            if len(group_data)<8:

                _models[dataset][group_name]=None

            else: 

                search = GridSearchCV(Model[Regressor], param_grid= {key : value for key, value in hyper_params[Regressor].items()})

                _ydata_group = group_data['outcome']
                _Xdata_group = Preprocessor.fit_transform(group_data)

                if Regressor == 'elasticnet':
                    # drop any column that has a missing value column in at least one group
                    for column in find_nan_columns(grouping_column)[dataset]:
                        if column in _Xdata_group.columns:
                            _Xdata_group.drop(columns=column, inplace=True)

                search.fit(_Xdata_group, _ydata_group)


                _Xdata = Preprocessor.fit_transform(_Xydata[dataset])
                _ypredict = search.predict(_Xdata)

                # Extract coefficients and intercept from the fitted ElasticNet model
                _models[dataset][group_name]={
                    f'{grouping_column}': group_data[grouping_column].iloc[0],  # Extract the group
                    'y_predict': _ypredict,
                    'model': search.best_estimator_  # Store the entire fitted model
                }

    #save_data(path=f'parameters_{Regressor}_{grouping_column}_data.parquet',results=_models)

#datasets=['eicu','hirid','mimic','miiv']

#_models_file=data.load_data(f'parameters_{Regressor}_{grouping_column}')
_dfmodels = pd.DataFrame(_models)

results_sources = {'eicu': {},
                    'hirid': {},
                    'mimic': {},
                    'miiv': {}}

for source in datasets: 

    if age_group:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[source]['age_group'] = pd.cut(_Xydata[source]['age'], bins=bins, labels=labels, right=False)


    if len(_Xydata[source].groupby(grouping_column))>1:

        if age_group: 
            _Xydata[source].drop(columns='age_group',inplace=True)
        
        _predictions=_dfmodels[source].dropna().values # All groups with no calculated parameters are skipped

        results=[] 

        _Xdata = Preprocessor.fit_transform(_Xydata[source])

        for column in find_nan_columns(grouping_column)[source]:
            if column in _Xdata.columns:
                _Xdata.drop(columns=column, inplace=True)

        r=len(_predictions)
        print([_predictions[i][grouping_column] for i in range(len(_predictions))])
        fhat = np.column_stack([(_predictions[i])['y_predict'] for i in range(len(_predictions))])
        print(fhat)
        H = fhat.T @ fhat / _Xdata.shape[0]
        print(H)

        if not all(np.linalg.eigvals(H) > 0): # H needs to be positive definite
            print("Attention: Matrix H is not positive definite")
            H += 1e-5
        
        P = matrix(H)
        q = matrix(np.zeros(r))
        G = matrix(-np.eye(r))
        h = matrix(np.zeros(r))
        A = matrix(1.0, (1, r))
        b = matrix(1.0)

        solution = solvers.qp(P, q, G, h, A, b)
        w = np.array(solution['x']).round(decimals=2).flatten() # magging weights
        print(w)

        for target in datasets:

            print(target)

            _Xtarget = Preprocessor.fit_transform(_Xydata[target])

            if Regressor == 'elasticnet':
                for column in find_nan_columns(grouping_column)[source]:
                    if column in _Xtarget.columns:
                        _Xtarget.drop(columns=column, inplace=True)

            # Make predictions and calculate MSE
            loaded_model_preds = []

            for i in range(len(_predictions)):
                # Load the stored model for this group
                loaded_model = _predictions[i]['model']

                # Calculate predictions using the loaded model
                y_pred_loaded = loaded_model.predict(_Xtarget)

                # Add the predictions to the list, weighted by the corresponding weights
                loaded_model_preds.append(w[i] * y_pred_loaded)

            # Calculate the final prediction by summing up the weighted predictions
            y_pred = np.sum(loaded_model_preds, axis=0)

            _ydata = _Xydata[target]['outcome']

            mse = mean_squared_error(_ydata, y_pred)

            # Store results
            results.append({
                'target': target,
                'mse': round(mse,2)
            })

        results_sources[source] = results

        print(f'Feature coefficients from {source}')
        #print(pd.DataFrame(results)[['weights']].to_string())
        print(pd.DataFrame(results).to_latex())
        print()

save_data(path=f'magging_{Regressor}_{grouping_column}_results.parquet',results=results_sources)

'''

datasets=['eicu','hirid','mimic','miiv']

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
_params_model=data.load_data(f'parameters_{Regressor}') 

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
    data = Preprocessor.fit_transform(_Xydata[target])
    
    # Handle combinations of groups
    for r in range(1, len(datasets)):
        for group_combination in combinations(datasets, r):
            if target not in group_combination:
                # Calculate weights for the selected groups
                theta = np.column_stack([_params[group]['parameters'] for group in group_combination])
                _Xdata = pd.concat([Preprocessor.fit_transform(_Xydata[group]) for group in group_combination])
                # Convert bool dtypes to float dtypes for matrix operations
                s = _Xdata.select_dtypes(include='bool').columns
                _Xdata[s] = _Xdata[s].astype('int')
                hatS = _Xdata.T @ _Xdata / _Xdata.shape[0]
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
                model_intercept = np.sum([w[i] * _params[group]['intercept'] for i, group in enumerate(group_combination)], axis=0)
                Model[Regressor].coef_ = model_coef
                Model[Regressor].intercept_ = model_intercept
                
                # Make predictions and calculate MSE
                y_pred = Model[Regressor].predict(data)
                mse = mean_squared_error(_Xydata[target]['outcome'], y_pred)
                
                # Store results in the error list
                error.append({
                    'target': target,
                    'group 1': group_combination[0] if len(group_combination) > 0 else '',
                    'group 2': group_combination[1] if len(group_combination) > 1 else '',
                    'group 3': group_combination[2] if len(group_combination) > 2 else '',
                    'mse': round(mse,2),
                    'weights': w
                })

print(pd.DataFrame(error).to_latex())
print()

print("Script run successful") '''