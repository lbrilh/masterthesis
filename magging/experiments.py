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
import magging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

Regressor='lasso'
grouping_column = 'numbedscategory'
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

Model={
    'lasso':Lasso(fit_intercept=True, max_iter=10000),
    'lgbm': LGBMRegressor(),
}

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False, lgbm=False)
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

# Specify the dataset you want to create the Tukey-Anscombe plots for
dataset_to_plot = 'eicu'

# Load the parameters
parameters = pd.read_parquet(parameters_file)
'''for i in range(len(parameters)): ### weird names
    residues = np.array(parameters[dataset_to_plot][i]['residues'])
    # Calculate the standard deviation of the residuals
    std_residuals = np.sqrt(np.abs(residues / np.std(residues)))

    # Create a plot of standardized residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(std_residuals)), std_residuals, marker='o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Standardized Residuals Plot")
    plt.xlabel("Observation Index")
    plt.ylabel("Standardized Residuals")
    plt.grid(True)
    plt.show()
raise ValueError
for i in range(len(parameters)):
    residues = np.array(parameters[dataset_to_plot][i]['residues'])
    # Calculate Cook's distances
    n = len(residues)
    model = sm.OLS(residues, np.ones(n)).fit()
    influence = model.get_influence()
    cooks_distance = influence.cooks_distance[0]

    # Create a Cook's distance plot
    plt.figure(figsize=(8, 6))
    plt.stem(cooks_distance, markerfmt='ro', linefmt='-', basefmt=' ')
    plt.title("Cook's Distance Plot")
    plt.xlabel("Observation Index")
    plt.ylabel("Cook's Distance")
    plt.grid(True)
    plt.show()

raise ValueError'''

# Create a list to store the QQ plot for each parameter's residues
plt.figure(figsize=(10, 5))

for i in range(len(parameters)):
    residues = np.array(parameters[dataset_to_plot][i]['residues'])
    
    # Create the QQ plot
    sm.qqplot(residues, line='s')
    
    plt.title(f'QQ Plot for Parameter {i} - {dataset_to_plot}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True)
    
    # Show the QQ plot for this parameter
    plt.show()

#raise ValueError


################## Tukey Anscombe Plot
# Iterate through each group and create a Tukey-Anscombe plot
for i in range(len(parameters)):
    # Create a list to store the Tukey-Anscombe values for this group
    tukey_anscombe_values = []
    

    residues = np.array(parameters[dataset_to_plot][i]['residues'])
    tukey_anscombe = (residues - np.mean(residues)) / np.std(residues)
    tukey_anscombe_values.append(tukey_anscombe)

    group = parameters[dataset_to_plot][i][grouping_column]

    # Create a Tukey-Anscombe plot for this group
    plt.figure(figsize=(10, 5))
    for i, tukey_anscombe in enumerate(tukey_anscombe_values):
        plt.scatter(np.arange(len(tukey_anscombe)), tukey_anscombe, label=f'Parameter {i}')
    plt.title(f'Tukey-Anscombe Plot for {dataset_to_plot}, Group {group}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Tukey-Anscombe Value')
    plt.legend()
    plt.grid(True)

    # Show the plot for this group
    plt.show()


#raise ValueError

#print(parameters_file)
parameters = pd.read_parquet(parameters_file)
#print(parameters['eicu'][0])

datasets = ['eicu','mimic','miiv','hirid']

list_of_vectors=[]
weights = magging.weights(_Xydata['eicu'], 'eicu', pipeline, parameters_file)

for i in range(len(parameters)):
    print(parameters['eicu'][i]['best_params'])
    print(np.linalg.norm(np.array(parameters['eicu'][i]['estimated_coeff']),1))

#raise ValueError
X = Preprocessor.fit_transform(_Xydata['eicu'])

maximin = weights[0]*parameters['eicu'][0]['estimated_coeff']

for i in range(1,len(parameters)):
    maximin += weights[i]*parameters['eicu'][i]['estimated_coeff']


sigma = (X.T@X)/X.shape[0]

print('Maximin distance: ')

print(np.array(maximin).T @ sigma @ np.array(maximin))

for i in range(len(parameters)):
    print(np.array(parameters['eicu'][i]['estimated_coeff']).T @ sigma @ np.array(parameters['eicu'][i]['estimated_coeff']))


raise ValueError
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