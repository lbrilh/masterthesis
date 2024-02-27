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

data = load_data_for_prediction(outcome=outcome)
_Xydata = {source: data[source]['train'][lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))] for source in ['eicu', 'mimic', 'miiv', 'hirid']}

_Xyall = pd.concat([_Xydata[source] for source in ['eicu', 'mimic', 'miiv', 'hirid']], ignore_index=True)

Preprocessing = ColumnTransformer(
    transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
    ).set_output(transform="pandas")

pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessing),
    ('model', LGBMRegressor(boosting_type='rf', feature_fraction=0.8))
])

fig, axs = plt.subplots(2,2, figsize=(15,9))

for i, source in enumerate(['eicu', 'mimic', 'miiv', 'hirid', 'all']): 

    if source == 'all': 
        plt.tight_layout()
        plt.savefig('images/sources_feature_importances_rf.png')
        plt.show()
        pipeline.fit(_Xyall, _Xyall['outcome'])
        plt.title(f"Target: {outcome}", fontweight='bold', fontsize=15)

        feature_importances = pd.DataFrame({'Importance': pipeline.named_steps['model'].feature_importances_, 
                                            'Feature Names': pipeline.named_steps['model'].feature_name_})
        
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances = feature_importances.iloc[:10]

        sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature Names'], orient='h')
    
    else: 
        row, col = divmod(i, 2)
        ax = axs[row, col]
        pipeline.fit(_Xydata[source], _Xydata[source]['outcome'])

        feature_importances = pd.DataFrame({'Importance': pipeline.named_steps['model'].feature_importances_, 
                                            'Feature Names': pipeline.named_steps['model'].feature_name_})
        
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances = feature_importances.iloc[:10]

        ax.set_title(f"Feature Importances - {source}")
        sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature Names'], orient='h', ax=ax)

plt.tight_layout()
plt.savefig('images/all_feature_importances_rf.png')
plt.show()