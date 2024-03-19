'''
    This code calculates and plots feature importances of predictor variables when using outcome as response variable.
    The feature importances are calculated using LightGBMs version of RandomForest.
    They are displayed for each datasource seperately and once when caluclated for all datasources together.
'''
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from preprocessing import make_feature_preprocessing
from constants import CATEGORICAL_COLUMNS
from icu_experiments.load_data import load_data_for_prediction

outcome = 'hr'

datasets = ['eicu', 'mimic', 'miiv', 'hirid']
data = load_data_for_prediction(outcome=outcome)
Xy_data = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in datasets}

preprocessing = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False) # no one-hot encoding of categorical variables
    ).set_output(transform="pandas")

X_all = pd.concat([preprocessing.fit_transform(Xy_data[source]) for source in datasets], ignore_index=True)
y_all = pd.concat([Xy_data[source]['outcome'] for source in datasets])

fig, axs = plt.subplots(2,2, figsize=(12,9))
for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid']):     
    row, col = divmod(i, 2)
    ax = axs[row, col]
    model = LGBMRegressor(boosting_type='rf', feature_fraction=0.8)
    model.fit(preprocessing.fit_transform(Xy_data[source]), Xy_data[source]['outcome'])
    feature_names = []
    for feature_name in model.feature_name_:
        if 'numeric' in feature_name:
            feature_names.append(feature_name.split("numeric__")[1])
        else:
            feature_names.append(feature_name.split("categorical__sex")[1])        
    feature_importances = pd.DataFrame({'Importance': model.feature_importances_, 
                                        'Feature Names': feature_names})
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importances = feature_importances.iloc[:10]
    ax.set_title(source)
    sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature Names'], orient='h', ax=ax, alpha=0.7)
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig('images/barplots/RF_individual_feature_importances.png')
plt.show()

model = LGBMRegressor(boosting_type='rf', feature_fraction=0.8)
model.fit(X_all, y_all)
feature_names = []
for feature_name in model.feature_name_:
    if 'numeric' in feature_name:
        feature_names.append(feature_name.split("numeric__")[1])
    else:
        feature_names.append(feature_name.split("categorical__sex")[1])
feature_importances = pd.DataFrame({'Importance': model.feature_importances_, 
                                    'Feature Names': feature_names})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
feature_importances = feature_importances.iloc[:10]
sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature Names'], orient='h', alpha=0.7)
plt.ylabel("")
plt.tight_layout()
plt.savefig('images/barplots/RF_all_feature_importances.png')
plt.show()