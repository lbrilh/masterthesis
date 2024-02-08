''' Find the optimal penalty terms for the groups in magging.
ToDo: Find optinmal CV score (add as metric magging distance to groups). 
    What is plot_output? 
    What is PolynomialFeature? Use different metrics / check metrics. (max. explained var).
    Convex.py 
    plot fio02 histogram for children --> Violates normal assumption
    connection alpha lasso, dsl (it reduces pdim problem to 1dim problem)
    make empirical magging nice
    MSE on unseen datasets?
    plot diagnostics for groups with optimal alphas
    Can I apply the 1 se rule to get optimal alpha automatically?
Use different alpha for different groups (alpha from comp. stat)
Weird: Alpha big enough --> children magging distance suddenly 0  (delete fio02 and suddenly it works properly)
Only after scaling the shortest distance to convex hull?
'''


import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

sys.path.append(root_dir)

from preprocessing import make_feature_preprocessing
from constants import NUMERICAL_COLUMNS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from magging import Magging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Regressor='magging'
grouping_column = 'age_group'
age_group = True

parameters_file = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column, 'parameters.parquet')
estimators_folder = os.path.join(current_script_dir, 'estimators', Regressor, grouping_column)
parquet_folder = os.path.join(current_script_dir, 'Parquet', Regressor, grouping_column)

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
    for dataset in ['mimic', 'eicu', 'miiv', 'hirid']: 
        _Xydata[dataset]['age_group'] = _Xydata[dataset]['age_group'].cat.remove_unused_categories()

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(grouping_column, 'outcome', missing_indicator=False, categorical_indicator=True, lgbm=False)
    ).set_output(transform="pandas")


# Specify the dataset you want to create the Tukey-Anscombe plots for
dataset_to_plot = 'mimic'

Xy = Preprocessor.fit_transform(_Xydata[dataset_to_plot])
'''sns.kdeplot(Xy['numeric__fio2'])
plt.show()
raise ValueError'''

# dict to store the results
results = {group: {'alpha': [], 'l1_norm': [], 'num features': [], 'mse': []} for group in _Xydata[dataset_to_plot]['age_group'].cat.categories}

for alpha in np.linspace(start=0.1, stop=10, num=50): 
    pipeline = Pipeline(steps=[
        ('preprocessing', Preprocessor),
        ('model', Magging(Lasso, f'grouping_column__{grouping_column}', alpha = alpha, max_iter = 10000))
        ])
    pipeline.named_steps['model'].group_fit(Xy, 'outcome__outcome')
    X = Xy.drop(columns = ['outcome__outcome', f'grouping_column__{grouping_column}'])
    pipeline.named_steps['model'].group_predictions(X)
    model = pipeline.named_steps['model']
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        coef_l1_norm = np.linalg.norm(model.models[group].coef_, ord=1)
        mse = mean_squared_error(_ygroup,model.models[group].predict(_Xgroup))
        num_features = np.count_nonzero(model.models[group].coef_)
        results[group]['alpha'].append(round(alpha,2))
        results[group]['num features'].append(num_features)
        results[group]['mse'].append(mse)
        results[group]['l1_norm'].append(coef_l1_norm)
    
for group in _Xydata[dataset_to_plot][grouping_column].cat.categories:
    print(group)
    print(pd.DataFrame(results[group]))

# L1 norm vs. penalty term plot
fig1, ax1 = plt.subplots()
if 'child' in _Xydata[dataset_to_plot][grouping_column].cat.categories:
    ax1.plot(results['child']['alpha'], results['child']['l1_norm'], label='child', color=(0.48942421, 0.72854938, 0.56751036), alpha=0.6 , linewidth=3)
ax1.plot(results['young adults']['alpha'], results['young adults']['l1_norm'], label='young adults', color=(0.24929311, 0.56486397, 0.5586654), alpha=0.6, linewidth=3)
ax1.plot(results['middle age']['alpha'], results['middle age']['l1_norm'], label='middle age', color=(0.11131735, 0.39155635, 0.53422678), alpha=0.6, linewidth=3)
ax1.plot(results['senior']['alpha'], results['senior']['l1_norm'], label='senior', color=(0.14573579, 0.29354139, 0.49847009), alpha=0.6, linewidth=3)
plt.xlabel('alpha')
plt.ylabel('L1 norm')
plt.title('L1 norm of coefficient vector vs penalty term', fontdict={'fontweight': 'bold', 'fontsize': 15})
ax1.legend()

# MSE vs. penalty term plot
fig2, ax2 = plt.subplots()
if 'child' in _Xydata[dataset_to_plot][grouping_column].cat.categories:
    ax2.plot(results['child']['alpha'], results['child']['mse'], label='child', color=(0.48942421, 0.72854938, 0.56751036), alpha=0.6, linewidth=3)
ax2.plot(results['young adults']['alpha'], results['young adults']['mse'], label='young adults', color=(0.24929311, 0.56486397, 0.5586654), alpha=0.6, linewidth=3)
ax2.plot(results['middle age']['alpha'], results['middle age']['mse'], label='middle age', color=(0.11131735, 0.39155635, 0.53422678), alpha=0.6, linewidth=3)
ax2.plot(results['senior']['alpha'], results['senior']['mse'], label='senior', color=(0.14573579, 0.29354139, 0.49847009), alpha=0.6, linewidth=3)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title('MSE of coefficient vector vs penalty term', fontdict={'fontweight': 'bold', 'fontsize': 15})
ax2.legend()

# Number of non-zero features in parameter vector vs penalty term plot
fig3, ax3 = plt.subplots()
if 'child' in _Xydata[dataset_to_plot][grouping_column].cat.categories: 
    ax3.plot(results['child']['alpha'], results['child']['num features'], label='child', color=(0.48942421, 0.72854938, 0.56751036), alpha=0.6, linewidth=3)
ax3.plot(results['young adults']['alpha'], results['young adults']['num features'], label='young adults', color=(0.24929311, 0.56486397, 0.5586654), alpha=0.6, linewidth=3)
ax3.plot(results['middle age']['alpha'], results['middle age']['num features'], label='middle age', color=(0.11131735, 0.39155635, 0.53422678), alpha=0.6, linewidth=3)
ax3.plot(results['senior']['alpha'], results['senior']['num features'], label='senior', color=(0.14573579, 0.29354139, 0.49847009), alpha=0.6, linewidth=3)
plt.xlabel('alpha')
plt.ylabel('Number of non-zero features')
plt.title('Number of non-zero features vs penalty term', fontdict={'fontweight': 'bold', 'fontsize': 15})
ax3.legend()
plt.show()