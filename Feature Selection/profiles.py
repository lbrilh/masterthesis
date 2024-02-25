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

# Function to adjust annotations to avoid overlap
def adjust_annotation_positions(ax, xs, ys, labels):
    positions = []
    last_y = None
    for x, y, label in sorted(zip(xs, ys, labels), key=lambda k: k[1]):
        if last_y is not None and abs(y - last_y) < 0.05:  # Adjust threshold as needed
            y += 0.5  # Adjust spacing as needed
        ax.annotate(label, xy=(x, y), xytext=(10, 0),
                    textcoords="offset points", ha='left', va='center')
        last_y = y
        positions.append(y)
    return positions

for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid']): 

    row, col = divmod(i, 2)
    ax = axs[row, col]

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
        feature_indices = np.argsort(np.abs(coefs[:, -1]))[-3:]
    elif source == 'hirid':
        feature_indices = np.argsort(np.abs(coefs[:, -1]))[-2:]
    else:
        feature_indices = np.argsort(np.abs(coefs[:, -1]))[-4:]

    print(f'Four most important features on {source}: {[list(_Xtrain.columns)[i] for i in np.argsort(np.abs(coefs[:, -1]))[-4:]]}')
    # Assuming you have a way to map these indices to original feature names
    feature_names = [list(_Xtrain.columns)[i] for i in feature_indices]

    # Adjusted part to handle annotations
    xmax = max(xx)

    y_positions = [coefs[i, -1] for i in feature_indices]

    # Dynamically adjust and place annotations to avoid overlap
    adjust_annotation_positions(ax, [xmax] * len(feature_indices), y_positions, feature_names)

fig.suptitle(f"Target: {outcome}", fontweight='bold', fontsize=15)
plt.tight_layout()  # Adjust the layout
plt.show()