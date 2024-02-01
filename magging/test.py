import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

sys.path.append(root_dir)

from preprocessing import make_feature_preprocessing
from constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from magging import Magging
from Diagnostics import SqrtAbsStandardizedResid, CookDistance, QQPlot, TukeyAnscombe, CorrelationPlot
import pandas as pd
import numpy as np

Regressor='magging'
grouping_column = 'age_group'
age_group = True


parameters_file = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet')
estimators_folder = os.path.join(current_script_dir, 'estimators', Regressor, grouping_column)
parquet_folder = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column)

hyper_params={
    'lasso':{"alpha": [9.7, 10]},
    'lgbm': {
        'boosting_type': ['gbdt'],
        'num_leaves': [20, 30, 40],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]}
}

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

# Apply age grouping if needed
if age_group:
    for dataset in ['mimic', 'eicu', 'miiv', 'hirid']:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)
        _Xydata[dataset]['age_group'].dropna(inplace=True)
        print(_Xydata[dataset]['age_group'].isna().sum())


Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(grouping_column, 'outcome', missing_indicator=False, categorical_indicator=True, lgbm=False)
    ).set_output(transform="pandas")

pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessor),
    ('model', Magging(Lasso, f'grouping_column__{grouping_column}', alpha = 2., max_iter = 10000))
])

print(_Xydata['hirid'].groupby(by=['age_group'])['source'].count())
# Specify the dataset you want to create the Tukey-Anscombe plots for
dataset_to_plot = 'hirid'
Xy = Preprocessor.fit_transform(_Xydata[dataset_to_plot])
pipeline.named_steps['model'].group_fit(Xy, 'outcome__outcome')

X = Xy.drop(columns = ['outcome__outcome', f'grouping_column__{grouping_column}'])
pipeline.named_steps['model'].group_predictions(X)

pipeline.named_steps['model'].weights(X)
yhat = pipeline.named_steps['model'].predict(X)

CorrelationPlot(yhat, Xy['outcome__outcome'])

#QQPlot(pipeline.named_steps['model'], Xy)
#TukeyAnscombe(pipeline.named_steps['model'], Xy)
print("Script successful")