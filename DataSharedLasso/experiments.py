import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

sys.path.append(root_dir)

from preprocessing import make_feature_preprocessing
from constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import DataSharedLasso
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

Regressor='elasticnet'
grouping_column = 'numbedscategory'
age_group = True


parameters_file = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet')
estimators_folder = os.path.join(current_script_dir, 'estimators', Regressor, grouping_column)
parquet_folder = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column)

hyper_params={
    'elasticnet':{"alpha": [0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100],
                 "l1_ratio": [1]},
}

Model={
    'elasticnet':ElasticNet(fit_intercept=True),
}

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=True, categorical_indicator=True, lgbm=False)
    ).set_output(transform="pandas")

pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessor),
    ('model', Model[Regressor])
])

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

datasets=['eicu','hirid','mimic','miiv']

# Apply age grouping if needed
if age_group:
    for dataset in datasets:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)

if os.path.exists(parameters_file):
    loaded_data = pd.read_parquet(parameters_file, engine='pyarrow')
    _dfmodels = pd.DataFrame(loaded_data)
else: 
    """
    Calculate the feature parameters within each group and store the results.
    Create the necessary folders if they do not already exist.
    """
    if not os.path.exists(estimators_folder):
        os.makedirs(estimators_folder)

    if not os.path.exists(parquet_folder):
        os.makedirs(parquet_folder)

    # The parameter vector for each dataset
    _models = {dataset: {} for dataset in datasets}

    for dataset in datasets:
        Z, y_tilde=DataSharedLasso.create_augmented_data(_Xydata[dataset], pipeline, grouping_column)
        search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_parameters.items()})
        search.fit(Z, y_tilde)
        _models[dataset] = {
                f'{grouping_column}': group_data[grouping_column].iloc[0],
                'estimated_coeff': search.best_estimator_.named_steps['model'].coef_,
                'y_predict': _ypredict,
                'residues': _ydata_group - search.predict(group_data),
                'model': estimator_file_path  # Store the filename instead of the model object
            }