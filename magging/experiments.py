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
import magging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

'''
ToDo Section: 
- Look at predictive performance within groups
- Look at Bühlmann instructions from beginning
- implement maybe one or two here 
- Make GitHub look nice
- Shared Lasso
- Group DRO

Ideen: 
- Schau bei einzelnen Gruppen nach ob outlier o.Ä. vorliegen
- Schau histogram / box plots der einzelnen Gruppen an ob da ein shift zu erkennen ist --> ggf. Methode zum estimaten innerhalb der Gruppe anpassen

'''

Regressor='lgbm'
grouping_column = 'region'
age_group = True


parameters_file = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet')
estimators_folder = os.path.join(current_script_dir, 'estimators', Regressor, grouping_column)
parquet_folder = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column)

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

# Load data from global parquet folder 
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


# Define a custom function to find columns with all missing values
def columns_with_missing(group):
    return group.columns[group.isnull().all()]

# Group your data by 'ethnic' and apply the custom function to each group
result = _Xydata['eicu'].groupby(by='numbedscategory').apply(columns_with_missing)
print(result.to_list())

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
    weights = magging.weights(_Xydata[train_source], pipeline, parameters_file)
    for test_source in datasets:
        _ypred = magging.predict(weights, _Xydata[test_source], pipeline, parameters_file)
        _ydata = _Xydata[test_source]['outcome']
        mse = mean_squared_error(_ydata, _ypred)
        results.append({
            'target': test_source,
            'mse': round(mse,3)
        })
        print(pd.DataFrame(results).to_latex())
        print()

print("Script run successful")