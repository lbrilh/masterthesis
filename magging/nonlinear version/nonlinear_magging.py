import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

sys.path.append(root_dir)

from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from icu_experiments.constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from cvxopt import matrix, solvers
from sklearn.metrics import mean_squared_error
import re

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



######################### ALL Paths need to be adjusted

Regressor='lgbm'
grouping_column = 'region'
age_group = True

hyper_params={
    'lgbm': {
        'boosting_type': ['gbdt'],
        'num_leaves': [20, 30, 40],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]}
}


Model={
    'lgbm': LGBMRegressor(),
}


Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False, lgbm=True)
    ).set_output(transform="pandas")


# Load data function
def load_data(outcome, source, version='train'):
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    relative_path = os.path.join('..', '..', 'Parquet', f'{outcome}_data_{source}_{version}.parquet')
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

datasets=['eicu','hirid','mimic','miiv']

# Apply age grouping if needed
if age_group:
    for dataset in datasets:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)


if not os.path.exists(os.path.join(root_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet')):

    """

    Calculate the feature parameters within each group and store the results. 

    """

    if not os.path.exists(os.path.join(root_dir, 'estimators', Regressor, grouping_column)):
        os.makedirs(os.path.join(root_dir, 'estimators', Regressor, grouping_column))

    if not os.path.exists(os.path.join(root_dir, 'Parquet', Regressor, grouping_column)):
        os.makedirs(os.path.join(root_dir, 'Parquet', Regressor, grouping_column))


    # The parameter vector for each dataset
    _models = {dataset: {} for dataset in datasets}

    for dataset in datasets:

        grouped=_Xydata[dataset].groupby(by=grouping_column)

        for group_name, group_data in grouped:
            
            # Skip all groups with less than 8 observations
            if len(group_data)<8:
                _models[dataset][group_name]=None
            else: 

                search = GridSearchCV(Model[Regressor], param_grid= {key : value for key, value in hyper_params[Regressor].items()})
                _ydata_group = group_data['outcome']
                _Xdata_group = Preprocessor.fit_transform(group_data)

                search.fit(_Xdata_group, _ydata_group)

                _Xdata = Preprocessor.fit_transform(_Xydata[dataset])
                _ypredict = search.predict(_Xdata)

                sanitized_group_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(group_name))
                estimator_filename = f'best_estimator_{dataset}_{sanitized_group_name}.joblib'
                estimator_file_path = os.path.join(root_dir, 'estimators', Regressor, grouping_column, estimator_filename)
                joblib.dump(search.best_estimator_, estimator_file_path)

                _models[dataset][group_name] = {
                    f'{grouping_column}': group_data[grouping_column].iloc[0],
                    'y_predict': _ypredict,
                    'model': estimator_file_path  # Store the filename instead of the model object
                }

    # Convert _models to a DataFrame and save it as a Parquet file
    _dfmodels = pd.DataFrame(_models)
    _dfmodels.to_parquet(os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet'))


else: 
    loaded_data = pd.read_parquet(os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet'), engine='pyarrow')
    _dfmodels = pd.DataFrame(loaded_data)

results_sources = {'eicu': {},
                    'hirid': {},
                    'mimic': {},
                    'miiv': {}}

for source in datasets: 

    if len(_Xydata[source].groupby(grouping_column))>1: # do not do predictions on the entire dataset
        
        _predictions=_dfmodels[source].dropna().values # All groups with no calculated parameters are skipped

        results=[] 
        
        _Xdata = Preprocessor.fit_transform(_Xydata[source])

        r=len(_predictions)
        fhat = np.column_stack([(_predictions[i])['y_predict'] for i in range(len(_predictions))])
        H = fhat.T @ fhat / _Xdata.shape[0]

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

        print(f'feature coefficients from {source}:')
        print([_predictions[i][grouping_column] for i in range(len(_predictions))])
        print(w)

        for target in datasets:

            _Xtarget = Preprocessor.fit_transform(_Xydata[target])

            # Make predictions and calculate MSE
            loaded_model_preds = []

            for i in range(len(_predictions)):
                # Load the stored model for this group
                loaded_model = joblib.load(_predictions[i]['model'])

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

        #print(pd.DataFrame(results)[['weights']].to_string())
        print(pd.DataFrame(results).to_latex())
        print()

print("Script run successful")