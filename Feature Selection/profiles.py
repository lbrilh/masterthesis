''' This code calculates and plots the Lasso profiles of predictor variables when using outcome as response variable. 
    The names of the four most important features are included in the plot.
    A figure containing the profiles on each data source will be shown at the end. Moreover, the profiles are shown when we 
    use all data sources at once. 

    
Tue: 
Do preprocessing before DSL estimation. Can I do so (look in DSL paper).
understand code, interpret pictures, prepare for Malte (How are plots calculated??)

Wed: 
Look at the different targets (i.e which might be interesting when considering sepsis?). 

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
from sklearn.pipeline import Pipeline

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'hr'
method = 'lasso'

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

Preprocessor = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False)
    ).set_output(transform="pandas")

fig, axs = plt.subplots(2,2, figsize=(15,9))

for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid', 'all']): 

    if source == 'all': 
        fig.suptitle(f"Target: {outcome}", fontweight='bold', fontsize=15)
        plt.tight_layout()  # Adjust the layout
        plt.savefig(f"images/{source}_lasso_path_incl_sex")
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

    print(f"Computing regularization path using the LARS on {source} ...")
    alphas, active, coefs = lars_path(_Xtrain.values, _y.values, method=method, verbose=True)

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    
    ax.plot(xx, coefs.T)
    ymin, ymax = ax.get_ylim()
    ax.vlines(xx, ymin, ymax, linestyle="dashed", alpha=0.1)
    ax.set_xlabel("|coef| / max|coef|")
    ax.set_ylabel("Coefficients")
    ax.set_title(f"LASSO Path - {source}")
    ax.axis("tight")
    
    # Identify the indices of the four largest coefficients in the last step
    if source == 'eicu':    
        feature_indices = np.argsort(np.abs(coefs[:, -1]))[-4:]
    elif source == 'hirid':
        feature_indices = np.argsort(np.abs(coefs[:, -1]))[-4:]
    else:
        feature_indices = np.argsort(np.abs(coefs[:, -1]))[-4:]

    print(f'Four most important features on {source}: {[list(_Xtrain.columns)[i] for i in np.argsort(np.abs(coefs[:, -1]))[-4:]]}')
    # Assuming you have a way to map these indices to original feature names
    feature_names = [list(_Xtrain.columns)[i] for i in feature_indices]

    xmax = max(xx)
    
    for i, feature_index in enumerate(feature_indices):
        y_pos = coefs[feature_index, -1]

        if source == 'hirid':
            # Place annotation closer to the line's end
            if feature_names[i] == "categorical__sex_Male":
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, -5), 
                        textcoords="offset points", ha='left', va='center')
            elif feature_names[i] == "categorical__sex_Female":
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, +5), 
                        textcoords="offset points", ha='left', va='center')
            else:
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, 0), 
                        textcoords="offset points", ha='left', va='center')
                
        elif source == 'all':
            if feature_names[i] == "categorical__sex_Male":
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, +3), 
                        textcoords="offset points", ha='left', va='center')
            elif feature_names[i] == "categorical__sex_Female":
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, -3), 
                        textcoords="offset points", ha='left', va='center')
            else:
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, 0), 
                        textcoords="offset points", ha='left', va='center')
                
        
        else: 
            # Place annotation closer to the line's end
            if feature_names[i] == "categorical__sex_Male":
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, 5), 
                        textcoords="offset points", ha='left', va='center')
            elif feature_names[i] == "categorical__sex_Female":
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, -5), 
                        textcoords="offset points", ha='left', va='center')
            else:
                ax.annotate(feature_names[i], xy=(xmax, y_pos), xytext=(5, 0), 
                        textcoords="offset points", ha='left', va='center')

fig.suptitle(f"Target: {outcome}", fontweight='bold', fontsize=15)
plt.tight_layout()  # Adjust the layout
plt.savefig("images/data_sources_lasso_path_incl_sex.png")
plt.show()