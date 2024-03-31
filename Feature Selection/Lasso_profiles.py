''' 
    This code calculates and plots the Lasso profiles of predictor variables when using outcome as response variable. 
    The names of the four most important features are included in the plot.
    A figure containing the profiles on each data source will be shown at the end. Moreover, the profiles are shown when we 
    use all data sources at once.
    Running this code requires CATEGORICAL_COLUMNS = ['sex'] in constants.py
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import lars_path
from sklearn.compose import ColumnTransformer

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'hr'
method = 'lasso'

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].isin(['Male', 'Female']))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

# Print and plot Lasso profiles on each dataset individually
fig, axs = plt.subplots(2,2, figsize=(15,9))
for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid', 'all']): 
    if source == 'all': 
        plt.tight_layout()  # Adjust the layout
        plt.savefig(f"images/lasso profiles/{outcome}/Individual_Lasso_coef_path_{outcome}.png")
        plt.show()
        plt.close()
        fig, ax = plt.subplots(figsize=(15,9))
    else: 
        row, col = divmod(i, 2)
        ax = axs[row, col]

    if source == 'all': 
        _Xytrain = pd.concat([_Xydata[source] for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)
    else: 
        _Xytrain = _Xydata[source]
    
    _y = _Xytrain["outcome"]
    _Xtrain = Preprocessor.fit_transform(_Xytrain)
    y_mean = _y.mean()

    print(f"Computing regularization path using the LARS on {source} ...")
    alphas, active, coefs = lars_path(_Xtrain.to_numpy(), _y.to_numpy()-y_mean, method=method, verbose=True)
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    ax.plot(xx, coefs.T)
    ymin, ymax = ax.get_ylim()
    ax.vlines(xx, ymin, ymax, linestyle="dashed", alpha=0.1)
    if source != 'all':
        ax.set_title(source, fontsize=16)
    if source == 'mimic':
        ax.set_yticks(ticks=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
    if source == 'all':
        ax.set_xlabel("|coef| / max|coef|", fontsize=20)
        ax.set_ylabel("Coefficients", fontsize=20)
        ax.tick_params(axis='y', labelsize=18, size=0) 
        ax.tick_params(axis='x', labelsize=18, size=0) 
    else: 
        ax.set_xlabel("|coef| / max|coef|", fontsize=15)
        ax.set_ylabel("Coefficients", fontsize=15)
        ax.tick_params(axis='y', labelsize=13, size=0) 
        ax.tick_params(axis='x', labelsize=13, size=0) 
    ax.axis("tight")

    print(f'Five most important features on {source}: {[list(_Xtrain.columns)[i] for i in np.argsort(np.abs(coefs[:, -1]))[-6:]]}')
    print()

    xmax = max(xx)

plt.tight_layout()  # Adjust the layout
plt.savefig(f"images/lasso profiles/{outcome}/Pooled_Lasso_coef_path_{outcome}.png")