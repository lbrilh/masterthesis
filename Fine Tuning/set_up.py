'''
    Set up parameters and configurations for experiments_script.py
'''

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge

from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from ivmodels.anchor_regression import AnchorRegression
from methods import AnchorBoost

overwrite = False
model = 'lgbm'
outcome = 'hr'
sources = ['eicu', 'hirid', 'mimic', 'miiv']
training_source = 'eicu'
anchor_columns = ['hospital_id']


n_seeds = 10 # number of seeds to use
n_fine_tuning = [100, 200, 400, 800, 1600] # number of fine-tuning data points 

boosting_methods=[
    'anchor_lgbm'
]

# Initialize the hyperparameter grid based on the model
hyper_parameters = {
    'ols': None,
    'ridge': {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        },
    'lgbm': {
        'boosting_type': ['gbdt'],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 800],
        'num_leaves': [50, 200, 1024],
        'feature_fraction': [0.5, 0.9],
        'verbose': [-1]
        },
    'rf': {
        'boosting_type': ['rf'],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 800],
        'num_leaves': [50, 200, 1024],
        'feature_fraction': [0.5, 0.9],
        'verbose': [-1]
        },
    'anchor': {
        'gamma': [1, 3.16, 10, 31.6, 100, 316, 1000, 3162, 10000],
        'instrument_regex': ['anchor'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'l1_ratio': [0, 0.2, 0.5, 0.8, 1]
    },
    'lgbm_refit':{
        "decay_rate": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        "cv": [0]
        },
    'anchor_lgbm': {
        'anchor': {
            'gamma': [1, 10, 100],
            'instrument_regex': ['anchor'],
            'alpha': [0.0001, 0.01],
            'l1_ratio': [0, 0.5]
        },
        'lgbm': {
            'boosting_type': ['gbdt'],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 800],
            'num_leaves': [50, 200, 1024],
            'feature_fraction': [0.5, 0.9],
            'verbose': [-1]
        }
    }
}

# Initialize the regressor and preprocessor based on the model
if model=='ols':
    regressor=LinearRegression()
    preprocessing=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
elif model=='ridge':
    regressor=Ridge()
    preprocessing=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
elif model=='lgbm' or model=='rf' or model=='lgbm_refit':
    regressor=LGBMRegressor()
    preprocessing=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
                                    ).set_output(transform="pandas")
elif model=='anchor':
    regressor=AnchorRegression()
    preprocessing=ColumnTransformer(transformers=
                                    make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
elif model=='anchor_lgbm':
    regressor=AnchorBoost()
    preprocessing=ColumnTransformer(transformers=
                                    make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")