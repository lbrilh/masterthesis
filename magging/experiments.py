import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

sys.path.append(root_dir)

from preprocessing import make_feature_preprocessing
from constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import magging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

Regressor='lasso'
grouping_column = 'age_group'
age_group = True


parameters_file = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet')
estimators_folder = os.path.join(current_script_dir, 'estimators', Regressor, grouping_column)
parquet_folder = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column)

alpha_grid = np.linspace(start=0, stop=10, num=50)

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(grouping_column, 'outcome', missing_indicator=False, categorical_indicator=True, lgbm=False)
    ).set_output(transform="pandas")


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
    for dataset in ['mimic', 'eicu', 'miiv', 'hirid']:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)
        _Xydata[dataset]['age_group'].dropna(inplace=True)

_Xydata['mimic'][grouping_column] = _Xydata['mimic'][grouping_column].cat.remove_unused_categories()

mse_results = {f'mse score in {group}': {} for group in _Xydata['mimic'][grouping_column].cat.categories}
Xy = Preprocessor.fit_transform(_Xydata['mimic'])
print(Xy.columns)

for group in _Xydata['mimic'][grouping_column].cat.categories:
    print(group)
    Xy = Xy[Xy[f'grouping_column__{grouping_column}']==group]
    X = Xy.drop(columns = ['outcome__outcome', f'grouping_column__{grouping_column}'])

    for alpha in alpha_grid:
        Lasso(fit_intercept=True, max_iter=10000, alpha=alpha)
        Lasso.fit(X, Xy[['outcome__outcome']])
        ypred = Lasso.predict(X)
    mse_results[f'mse score in {group}'] = mean_squared_error(Xy['outcome'],ypred)

print(pd.DataFrame(mse_results))

raise ValueError




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
        _models[dataset]=magging.group_predictions(_Xydata[dataset], dataset, grouping_column, hyper_params[Regressor], pipeline, estimators_folder)

    _dfmodels = pd.DataFrame(_models)
    _dfmodels.to_parquet(parameters_file)

# Calculate the magging Mean-Squared Error
for train_source in datasets:
    results = []
    print('Currently on: ', train_source)
    weights = magging.weights(_Xydata[train_source], train_source, pipeline, parameters_file)
    for test_source in datasets:
        _ypred = magging.predict(weights, _Xydata[test_source], pipeline, parameters_file, train_source)
        _ydata = _Xydata[test_source]['outcome']
        mse = mean_squared_error(_ydata, _ypred)
        results.append({
            'target': test_source,
            'mse': round(mse,3)
        })
        print(pd.DataFrame(results).to_latex())
        print()

print("Script run successful")