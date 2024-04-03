'''
    This code calculates and plots the Data Shared Lasso profiles of predictor variables when using outcome as response variable.
    The profiles of the shared effects are plotted and sepearetly the profiles of the group estimates.
    Running this code requires CATEGORICAL_COLUMNS = ['sex', 'source'] in constants.py
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import lars_path
from sklearn.pipeline import Pipeline

from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction
from preprocessing import make_feature_preprocessing

outcome = 'hr'
method = 'lasso'

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

# Preprocessing of individual groups
Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

Xy_train = pd.concat([_Xydata[source] for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
X_train = pd.concat([Preprocessor.fit_transform(_Xydata[source]) for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
X_train.fillna(0, inplace=True)
y_train = Xy_train['outcome']
y_mean = y_train.mean()
y_train = y_train - y_mean

# Start generating the augmented dataset
interactions = ColumnTransformer(
    [
        (
            "passthrough_features",
            "passthrough",
            [
                column
                for column in X_train.columns
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
                                for source in X_train.columns
                                if "categorical__source" in source
                                for column in X_train.columns
                                if "categorical__source" not in column
                            ]
            ),
            [column for column in X_train.columns],
        )
    ]
).set_output(transform="pandas")

X_train_augmented = interactions.fit_transform(X_train) 

# Control degree of sharing
r_g = {source: len(_Xydata[source])/len(X_train) for source in ['eicu', 'mimic', 'miiv', 'hirid']}

for column in X_train_augmented.columns: 
    if 'passthrough' not in column: 
        if 'eicu' in column: 
            X_train_augmented[column] = 1/np.sqrt(r_g['eicu'])*X_train_augmented[column]
        elif 'mimic' in column:
            X_train_augmented[column] = 1/np.sqrt(r_g['mimic'])*X_train_augmented[column]
        elif 'miiv' in column:
            X_train_augmented[column] = 1/np.sqrt(r_g['miiv'])*X_train_augmented[column]
        elif 'hirid' in column: 
            X_train_augmented[column] = 1/np.sqrt(r_g['hirid'])*X_train_augmented[column]
    
# Calculate the Lasso profiles
print(f"Computing regularization path using LARS on DSL ...")
alphas, active, coefs = lars_path(X_train_augmented.to_numpy(), y_train.to_numpy(), method=method, verbose=True)

# Print and plot Lasso path of common effects against shrinkage factor
coefs_shared = coefs[:51, :] # shared effects
fig, ax = plt.subplots(figsize=(15,9))
xx = np.sum(np.abs(coefs_shared.T), axis=1)
xx /= xx[-1]
feature_indices = np.argsort(np.abs(coefs_shared[:, -1]))[-4:]
feature_names = [list(X_train_augmented.columns)[i] for i in feature_indices]
xmax = max(xx)
print('common effects: ', feature_names)
print()
ax.plot(xx, coefs_shared.T)
ymin, ymax = ax.get_ylim()
ax.vlines(xx, ymin, ymax, linestyle="dashed", alpha=0.1)
ax.set_xlabel("|coef| / max|coef|", fontsize=20)
ax.set_ylabel("Coefficients", fontsize=20)
ax.tick_params(axis='y', labelsize=18, size=0) 
ax.tick_params(axis='x', labelsize=18, size=0)  
ax.axis("tight")
plt.tight_layout()  # Adjust the layout
plt.savefig(f'images/lasso profiles/{outcome}/{outcome}_common_effects_DSL_preprocessed_individ.png')
plt.show()


r_g = {'eicu': 1.41, 'hirid': 4.17, 'mimic': 2.22, 'miiv': 2.05}
# Print and plot Lasso path of group effects against shrinkage factor
fig, axs = plt.subplots(2, 2, figsize=(15,9))
for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid']): 
    _coefs = coefs[(1+i)*51:(2+i)*51, :]
    row, col = divmod(i, 2)
    ax = axs[row, col]
    xx = np.sum(np.abs(_coefs.T), axis=1)
    xx /= xx[-1]
    feature_indices = np.argsort(np.abs(_coefs[:, -1]))[-4:]
    feature_names = [list(X_train_augmented.iloc[:, (1+i)*51:(2+i)*51].columns)[j] for j in feature_indices]
    xmax = max(xx)
    print(f'{source} effects: ', feature_names)
    ax.plot(xx, r_g[source]*_coefs.T)
    ymin, ymax = ax.get_ylim()
    ax.vlines(xx, ymin, ymax, linestyle="dashed", alpha=0.1)
    ax.set_xlabel("|coef| / max|coef|", fontsize=15)
    ax.set_ylabel("Coefficients", fontsize=15)
    ax.tick_params(axis='y', labelsize=13, size=0) 
    ax.tick_params(axis='x', labelsize=13, size=0)  
    ax.axis("tight")
    ax.set_title(source, fontsize=16)
plt.tight_layout()  # Adjust the layout
plt.savefig(f'images/lasso profiles/{outcome}/{outcome}_individual_groups_DSL_preprocessed_individ.png')
plt.show()