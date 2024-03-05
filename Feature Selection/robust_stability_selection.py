# TODO change marker color according to group
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'hr'

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

coefs = pd.read_parquet("dsl_coefs.parquet")
intercept = ((pd.read_parquet("dsl_intercepts_random_states.parquet")).mean(axis=1))[0]

# Preprocessing of individual groups
preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

_Xytrain = pd.concat([_Xydata[source] for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
_Xtrain = pd.concat([preprocessor.fit_transform(_Xydata[source]) for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
_Xtrain.fillna(0, inplace=True)
_ytrain = _Xytrain['outcome']

# Start generating the augmented dataset
interactions = ColumnTransformer(
    transformers = 
    [
        (
            "passthrough_features",
            "passthrough",
            [
                column
                for column in _Xtrain.columns
                if "categorical__source" not in column
            ],
        ),
        (
            "interactions",
            ColumnTransformer(
                            [
                                (
                                    f"{source}_{column}".replace('categorical__source_', '').replace("__", "_"),
                                    PolynomialFeatures(
                                        degree=(2, 2),
                                        include_bias=False,
                                        interaction_only=True,
                                    ),
                                    [source, column],
                                )
                                for source in _Xtrain.columns
                                if "categorical__source" in source
                                for column in _Xtrain.columns
                                if "categorical__source" not in column
                            ]
            ),
            [column for column in _Xtrain.columns],
        )
    ]
).set_output(transform="pandas")

_Xtrain_augmented = interactions.fit_transform(_Xtrain) 

# Control degree of sharing
r_g = {source: len(_Xydata[source])/len(_Xtrain) for source in ['eicu', 'mimic', 'miiv', 'hirid']}

for column in _Xtrain_augmented.columns: 
    if 'passthrough' not in column: 
        if 'eicu' in column: 
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['eicu'])*_Xtrain_augmented[column]
        elif 'mimic' in column:
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['mimic'])*_Xtrain_augmented[column]
        elif 'miiv' in column:
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['miiv'])*_Xtrain_augmented[column]
        elif 'hirid' in column: 
            _Xtrain_augmented[column] = 1/np.sqrt(r_g['hirid'])*_Xtrain_augmented[column]

mse = {}
mse['intercept'] = [mean_squared_error(np.repeat(intercept, len(_ytrain)), _ytrain)]

num_feat = 1
for feature in coefs['feature names']:
    X_np = (_Xtrain_augmented[coefs['feature names'][:num_feat]]).to_numpy()
    coefs_np = coefs['Avg coefs'][:num_feat].to_numpy()
    y_pred = intercept + X_np@coefs_np
    mse[f'{num_feat} feat.'] = [mean_squared_error(_ytrain, y_pred)]
    num_feat += 1

results = pd.DataFrame(mse)
pd.DataFrame(mse).to_parquet('robust_stability_selection.parquet')

fig, axs = plt.subplots(2,1,figsize=(12,9))
ax = axs[0]
ax.set_ylabel('Mean-Squared Error')
ax.set_xlabel('Number of features')
ax.grid(visible=True)
ax.plot(range(0,results.shape[1]), results.to_numpy().reshape(-1), 'bo-', ms=5, alpha=0.5)

gx = axs[1]
gx.set_ylabel('Mean-Squared Error')
gx.set_xlabel('Number of features')
gx.plot(range(0,50), results.to_numpy().reshape(-1)[:50], 'bo-', ms=5, alpha=0.5)
gx.grid(True)
plt.tight_layout()

fig2, axs2 = plt.subplots(2,2,figsize=(12,9))
mse_datasets = {source: {} for source in ['eicu', 'mimic', 'miiv', 'hirid']}
n_start = 0
for i, dataset in enumerate(['eicu', 'mimic', 'miiv', 'hirid']):
    row, col = divmod(i, 2)
    fx = axs2[row, col]
    nrows = n_start + _Xydata[dataset].shape[0]
    X_dataset = _Xtrain_augmented.iloc[n_start:nrows]
    y_dataset = _ytrain.iloc[n_start:nrows]
    mse_datasets[dataset]['intercept'] = [mean_squared_error(np.repeat(intercept, len(y_dataset)), y_dataset)]
    n_start += _Xydata[dataset].shape[0]
    feature_list = []
    for feature in coefs['feature names']:
        if any(x in feature for x in ['passthrough', dataset]):
            feature_list.append(feature)
            X_np = (X_dataset[feature_list]).to_numpy()
            coefs_np = coefs.loc[coefs['feature names'].isin(feature_list)]['Avg coefs'].to_numpy()
            y_pred = intercept + (X_np@coefs_np)
            mse_datasets[dataset][f'{len(feature_list)} feat.'] = [mean_squared_error(y_dataset, y_pred)]
    fx.set_title(dataset)
    fx.set_ylabel('Mean-Squared Error')
    fx.set_xlabel('Number of features')
    fx.grid(visible=True)
    fx.plot(range(0, len(feature_list)+1), pd.DataFrame(mse_datasets[dataset]).to_numpy().reshape((len(feature_list)+1,1)), 'bo-', alpha=0.5, ms=3)
fig2.suptitle(f"Individual", fontweight='bold', fontsize=15)
plt.tight_layout()
plt.show()

_distr_results = {f'Nr Groups {i}': {} for i in range(1,4)}
mse_comb_datasets = {f'Nr Groups {i}': {} for i in range(1,4)}
for r in range(1,4):
    mse_comb_datasets[f'Nr Groups {r}'] = {dataset: {} for dataset in ['eicu', 'mimic', 'miiv', 'hirid']}
    _distr_results[f'Nr Groups {r}'] = {dataset: {} for dataset in ['eicu', 'mimic', 'miiv', 'hirid']}
    for group_combination in combinations(['eicu', 'mimic', 'miiv', 'hirid'], r):
        group_list = list(group_combination)
        group_comb_name = ", ".join(group_list)
        n_start = 0
        for i, dataset in enumerate(['eicu', 'mimic', 'miiv', 'hirid']):
            if dataset not in group_combination: 
                nrows = n_start + _Xydata[dataset].shape[0]
                X_dataset = _Xtrain_augmented.iloc[n_start:nrows]
                y_dataset = _ytrain.iloc[n_start:nrows]
                mse_comb_datasets[f'Nr Groups {r}'][dataset][group_comb_name] = {}
                _distr_results[f'Nr Groups {r}'][dataset][group_comb_name] = {}
                mse_comb_datasets[f'Nr Groups {r}'][dataset][group_comb_name]['intercept'] = [mean_squared_error(np.repeat(intercept, len(y_dataset)), y_dataset)]
                n_start += _Xydata[dataset].shape[0]
                feature_list = []
                group_list.append('passthrough')
                for feature in coefs['feature names']:
                    if any(x in feature for x in group_list):
                        feature_list.append(feature)
                        X_np = (X_dataset[feature_list]).to_numpy() ########### not correct - need to select the data from the groups in group_combination; rn: select from target (i.e mostly 0)
                        coefs_np = coefs.loc[coefs['feature names'].isin(feature_list)]['Avg coefs'].to_numpy()
                        y_pred = intercept + (X_np@coefs_np)
                        mse_comb_datasets[f'Nr Groups {r}'][dataset][group_comb_name][f'{len(feature_list)} feat.'] = [mean_squared_error(y_dataset, y_pred)]
                _distr_results[f'Nr Groups {r}'][dataset][group_comb_name]['features'] = feature_list
    print(_distr_results[f'Nr Groups {r}'])
print(pd.DataFrame(mse_comb_datasets))
pd.DataFrame(mse_comb_datasets).to_parquet('mse_robust.parquet')
pd.DataFrame(_distr_results).to_parquet('dsl_feature_list.parquet')