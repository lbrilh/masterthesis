''' 
    Calculate the group estimators based on the alphas found in optimal_penalty.py
    In theory, preprocessing.py must be adjusted. 
'''
import os
import pandas as pd
import numpy as np
from Diagnostics import SqrtAbsStandardizedResid, CookDistance, QQPlot, TukeyAnscombe, CorrelationPlot
from magging import Magging
from cvxopt import matrix, solvers
from preprocessing import make_feature_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import statsmodels.api as sm


Regressor='magging'
grouping_column = 'age_group'
age_group = True
dataset_to_plot = 'mimic'
oracle_dataset = 'miiv'

optimal_alphas = {'child': 0.001, 'young adults': 0.001, 'middle age': 0.001, 'senior': 0.001}

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


preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(grouping_column, 'outcome', missing_indicator=False, categorical_indicator=True, lgbm=False)
    ).set_output(transform="pandas")

Xy = preprocessor.fit_transform(_Xydata[dataset_to_plot])
oracle_Xy = preprocessor.fit_transform(_Xydata[oracle_dataset])
oracle_X = oracle_Xy.drop(columns=['outcome__outcome', f'grouping_column__{grouping_column}'])
X = Xy.drop(columns = ['outcome__outcome', f'grouping_column__{grouping_column}'])

group_predictions = []
out_of_sample_prediction = []
in_group_pred = {}
for group in ['child', 'young adults', 'middle age', 'senior']:
    model = Lasso(max_iter=10000, alpha=optimal_alphas[group])
    _Xygroup = Xy[Xy[f'grouping_column__{grouping_column}'] == group]
    _Xgroup = _Xygroup.drop(columns=[f'grouping_column__{grouping_column}', 'outcome__outcome'])   
    _ygroup = np.array(_Xygroup[['outcome__outcome']]).reshape(-1)
    model.fit(_Xgroup, _ygroup)
    in_group_pred[group] = model.predict(_Xgroup)
    out_of_sample_prediction.append(model.predict(oracle_X))
    print(model.coef_)
    Sigma = (np.matrix(X).T@np.matrix(X))/np.matrix(X).shape[0]
    u_array = np.array(model.coef_)
    print(f'Magging distance for group {group}: ', ((u_array).T @ Sigma @ (u_array)))   
    group_predictions.append(model.predict(X))

group_prediction_matrix = np.matrix(group_predictions).T
r = group_prediction_matrix.shape[1]
if r == 1:
    print('Warning: Only one group exists!')
fhat = group_prediction_matrix
H = fhat.T @ fhat / oracle_X.shape[0]

print(np.linalg.eigvals(H))

if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
    print("Warning: Matrix H is not positive definite")
    H += 1e-5
P = matrix(H)
q = matrix(np.zeros(r))
G = matrix(-np.eye(r))
h = matrix(np.zeros(r))
A = matrix(1.0, (1, r))
b = matrix(1.0)

# Solve the quadratic program to obtain magging weights
solution = solvers.qp(P, q, G, h, A, b)
w = np.array(solution['x']).round(4).flatten() # Magging weights
print(w)

y_pred = np.dot(w, out_of_sample_prediction)
mse = mean_squared_error(oracle_Xy['outcome__outcome'].values, y_pred)
print(mse)

pipeline.named_steps['model'].group_predictions(X)

for group in pipeline.named_steps['model'].groups:
    print(group, pipeline.named_steps['model'].models[group].coef_)

pipeline.named_steps['model'].weights(X)
yhat = pipeline.named_steps['model'].predict(X)

CorrelationPlot(yhat, Xy['outcome__outcome'])

QQPlot(pipeline.named_steps['model'], Xy)
TukeyAnscombe(pipeline.named_steps['model'], Xy)
print("Script successful")