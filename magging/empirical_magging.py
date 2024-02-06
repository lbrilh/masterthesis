''' Calculate the group estimators based on the alphas found in optimal_penalty.py
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
optimal_alphas = {'child': 1.72, 'young adults': 2.73 , 'middle age': 2.32, 'senior': 1.92}

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
        ('model', Magging(Lasso, f'grouping_column__{grouping_column}', max_iter = 10000))
        ])

Xy = Preprocessor.fit_transform(_Xydata[dataset_to_plot])
X = Xy.drop(columns = ['outcome__outcome', f'grouping_column__{grouping_column}'])

group_predictions = []
in_group_pred = {}
for group in ['child', 'young adults', 'middle age', 'senior']:
    model = Lasso(max_iter=10000, alpha=optimal_alphas[group])
    _Xygroup = Xy[Xy[f'grouping_column__{grouping_column}'] == group]
    _Xgroup = _Xygroup.drop(columns=[f'grouping_column__{grouping_column}', 'outcome__outcome'])   
    _ygroup = np.array(_Xygroup[['outcome__outcome']]).reshape(-1)
    model.fit(_Xgroup, _ygroup)
    in_group_pred[group] = model.predict(_Xgroup)
    print(model.coef_)
    Sigma = (np.matrix(X).T@np.matrix(X))/np.matrix(X).shape[0]
    u_array = np.array(model.coef_)
    print(f'Magging distance for group {group}: ', ((u_array).T @ Sigma @ (u_array)))
    group_predictions.append(model.predict(X))

nrows = len(['child', 'young adults', 'middle age', 'senior'])//2
ncols = 2
# Standardized residuals plot
fig, axes = plt.subplots(nrows,ncols)
for i, group in enumerate(['child', 'young adults', 'middle age', 'senior']):
    _Xygroup = Xy[Xy[f'grouping_column__{grouping_column}'] == group]
    residues = np.array(_Xygroup['outcome__outcome'] - np.array(in_group_pred[group]))
    transformed_residuals = np.sqrt(np.abs(residues / np.std(residues)))
    ax = plt.subplot(nrows, ncols, i+1)
    plt.scatter(np.arange(len(transformed_residuals)), transformed_residuals, marker='o', alpha=0.05)
    ax.spines[['right','top']].set_visible(False)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.title(group)
fig.suptitle("Standardized Residuals Plot", fontweight='bold', fontsize=15)
plt.tight_layout()
#plt.show()

nrows = len(['child', 'young adults', 'middle age', 'senior'])//2
ncols = 2
# Tukey Anscombe plot
########################################## Center y axis
fig, axes = plt.subplots(nrows,ncols)
for i, group in enumerate(['child', 'young adults', 'middle age', 'senior']):
    _Xygroup = Xy[Xy[f'grouping_column__{grouping_column}'] == group]
    residues = np.array(_Xygroup['outcome__outcome'] - np.array(in_group_pred[group]))
    tukey_anscombe = (residues - np.mean(residues)) / np.std(residues)
    ax = plt.subplot(nrows, ncols, i+1)
    plt.scatter(np.arange(len(tukey_anscombe)), tukey_anscombe, alpha=0.05)
    plt.yticks([-5,0,5])
    plt.ylim([-5,5])
    ax.spines['bottom'].set_position('zero')
    ax.spines[['right','top']].set_visible(False)
    plt.title(group)
fig.suptitle("Tukey-Anscombe Plot for Groups", fontweight='bold', fontsize=15)
plt.tight_layout()
#plt.show()

# QQ Plot
nrows = len(['child', 'young adults', 'middle age', 'senior'])//2
ncols = 2
fig, axes = plt.subplots(nrows,ncols)
for i, group in enumerate(['child', 'young adults', 'middle age', 'senior']):
    _Xygroup = Xy[Xy[f'grouping_column__{grouping_column}'] == group]
    residues = np.array(_Xygroup['outcome__outcome'] - np.array(in_group_pred[group]))
    ax=plt.subplot(nrows, ncols, i+1)
    # Create the QQ plot
    sm.qqplot(residues, line='s', ax=ax)
    #sm.qqplot(residues, line='s', ax=axes.flatten()[i])
    plt.yticks([-100,0,100])
    plt.title(group)
plt.suptitle(f'QQ Plot', fontweight='bold', fontsize=15)
plt.tight_layout()
#plt.show()

# Correlation plot
for i, group in enumerate(['child', 'young adults', 'middle age', 'senior']):
    _Xygroup = Xy[Xy[f'grouping_column__{grouping_column}'] == group]
    plt.scatter(np.array(in_group_pred[group]), _Xygroup['outcome__outcome'], alpha=0.2)
    plt.title('Correlation of estimated and true y', fontweight='bold', fontsize=15)
    plt.xlabel('Predictions')
    plt.ylabel('y')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
#    plt.show()

group_prediction_matrix = np.matrix(group_predictions).T
r = group_prediction_matrix.shape[1]
if r == 1:
    print('Warning: Only one group exists!')
fhat = group_prediction_matrix
H = fhat.T @ fhat / X.shape[0]

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


raise ValueError
pipeline.named_steps['model'].group_predictions(X)

for group in pipeline.named_steps['model'].groups:
    print(group, pipeline.named_steps['model'].models[group].coef_)

raise ValueError
pipeline.named_steps['model'].weights(X)
yhat = pipeline.named_steps['model'].predict(X)

CorrelationPlot(yhat, Xy['outcome__outcome'])

QQPlot(pipeline.named_steps['model'], Xy)
TukeyAnscombe(pipeline.named_steps['model'], Xy)
print("Script successful")