'''
    This code calculates and plots the Data Shared Lasso profiles of predictor variables when using outcome as response variable.
    The profiles of the shared effects are plotted and sepearetly the profiles of the group estimates.
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import lars_path, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'map'
method = 'lasso'

data = load_data_for_prediction(outcome=outcome)

_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

_Xytrain = pd.concat([_Xydata[source] for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)

_ytrain = _Xytrain['outcome']
y_mean = _ytrain.mean()
_Xtrain = Preprocessor.fit_transform(_Xytrain)
'''
_Xtrain = pd.concat([Preprocessor.fit_transform(_Xydata[source]) for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
_Xtrain.fillna(0, inplace=True)'''

interactions = ColumnTransformer(
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
)

pipeline = Pipeline(
    steps=[("interactions", interactions)]
    ).set_output(transform="pandas")

_Xtrain_augmented = pipeline.fit_transform(_Xtrain) 

r = 1/np.sqrt(len(['eicu', 'mimic', 'miiv', 'hirid']))

for column in _Xtrain_augmented.columns: 
    if 'passthrough' not in column: 
        _Xtrain_augmented[column] = r*_Xtrain_augmented[column]
    


print(f"Computing regularization path using LARS on DSL ...")
alphas, active, coefs = lars_path(_Xtrain_augmented.to_numpy(), _ytrain.to_numpy() - y_mean, method=method, verbose=True)

coefs_shared = coefs[:51, :]

fig, ax = plt.subplots(figsize=(15,9))

xx = np.sum(np.abs(coefs_shared.T), axis=1)
xx /= xx[-1]

feature_indices = np.argsort(np.abs(coefs_shared[:, -1]))[-4:]
feature_names = [list(_Xtrain_augmented.columns)[i] for i in feature_indices]
xmax = max(xx)
    
print('common effects: ', feature_names)

ax.plot(xx, coefs_shared.T)
ymin, ymax = ax.get_ylim()
ax.vlines(xx, ymin, ymax, linestyle="dashed", alpha=0.1)
ax.set_xlabel("|coef| / max|coef|")
ax.set_ylabel("Coefficients")
ax.set_title(f"LASSO Path of shared coeffs in DSL")
ax.axis("tight")

fig.suptitle(f"Target: {outcome}", fontweight='bold', fontsize=15)
plt.tight_layout()  # Adjust the layout
plt.savefig('images/common_effects_DSL_preprocessed_individ.png')
plt.show()
plt.close()

fig, axs = plt.subplots(2, 2, figsize=(15,9))

for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid']): 

    _coefs = coefs[(1+i)*51:(2+i)*51, :]

    row, col = divmod(i, 2)
    ax = axs[row, col]

    xx = np.sum(np.abs(_coefs.T), axis=1)
    xx /= xx[-1]

    feature_indices = np.argsort(np.abs(_coefs[:, -1]))[-4:]
    feature_names = [list(_Xtrain_augmented.iloc[:, (1+i)*51:(2+i)*51].columns)[j] for j in feature_indices]
    xmax = max(xx)
        
    print(f'{source} effects: ', feature_names)

    ax.plot(xx, _coefs.T)
    ymin, ymax = ax.get_ylim()
    ax.vlines(xx, ymin, ymax, linestyle="dashed", alpha=0.1)
    ax.set_xlabel("|coef| / max|coef|")
    ax.set_ylabel("Coefficients")
    ax.set_title(f"LASSO Path of DSL on {source}")
    ax.axis("tight")

fig.suptitle(f"Target: {outcome}", fontweight='bold', fontsize=15)
plt.tight_layout()  # Adjust the layout
plt.savefig('images/individual_groups_DSL_preprocessed_individ.png')
plt.show()