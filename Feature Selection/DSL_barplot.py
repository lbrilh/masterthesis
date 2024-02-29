'''
    This code plots the 10 most important coefficients (absolute size) of DSL in a barplot.
    Positive coefficients are blue and negative coefficients are red. 
    The penalty term alpha is chosen via cross-validation. 
    The degree of sharing is controlled using r_g. The value of r_g is the relative size of the group to the entire dataset, i.e. larger groups are forced to 
    share more than smaller groups. Each group is preprocessed individually. 
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'hr'

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

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


# Find best penalization parameter
parameters = {'alpha': np.linspace(0.001, 10, 100)}
search = GridSearchCV(Lasso(max_iter=100000), parameters)
search.fit(_Xtrain_augmented, _ytrain)

# Extract coefficients and plot them 
coefs = search.best_estimator_.coef_
feat_names = search.best_estimator_.feature_names_in_
coef_df = pd.DataFrame({'feature names': feat_names, 'abs_coefs': np.abs(coefs), 'coefs': coefs})
coef_df['color'] = (coef_df['coefs']>=0).astype('bool')

color_palette = []
for color_indice in coef_df['color']:
    if color_indice == 1:
        color_palette.append('b')
    else: 
        color_palette.append('r')

coef_df.sort_values(by='abs_coefs', inplace=True, ascending=False)
names = ['hr', 'mimic sex male', 'mimic sex female', 'mimic hr', 'mimic height', 'temp', 'mimic age', 'na', 'sbp', 'eicu temp']

plt.figure(figsize=(12,9))
sns.barplot(x=coef_df["abs_coefs"].iloc[:10], y=names, orient="h", palette=color_palette[:10])
plt.xlabel("Absolute Value of Coefficient")
plt.tight_layout()
plt.savefig(f'images/DSL/Barplot_coefs_{outcome}.png')
plt.show()