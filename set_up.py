from icu_experiments.load_data import load_data_for_prediction
from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from lightgbm import LGBMRegressor
from ivmodels.anchor_regression import AnchorRegression
from anchor_boosting_lgbm import CustomizedAnchor



model = 'anchor'
outcome = 'hr'
n_seeds = 10    

sources = ['eicu', 'hirid', 'mimic', 'miiv']

training_source = 'eicu'

anchor_columns = ['hospital_id']

n_fine_tuning = [25, 50, 100, 200, 400, 800, 1600]


grid_search = False
evaluation_on_target = True

methods = [
    'ols',
    'ridge',
    'lgbm',
    'rf',
    'anchor',
    'anchor_boost'
]

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
    'anchor_boost': {
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

if model=='ols':
    Regressor=LinearRegression()
    Preprocessing=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
elif model=='ridge':
    Regressor=Ridge()
    Preprocessing=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
elif model=='lgbm' or model=='rf':
    Regressor=LGBMRegressor()
    Preprocessing=ColumnTransformer(transformers=
                                    make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
                                    ).set_output(transform="pandas"),
elif model=='anchor':
    Regressor=AnchorRegression()
    Preprocessing=ColumnTransformer(transformers=
                                    make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")
elif model=='anchor_boost':
    Regressor=CustomizedAnchor()
    Preprocessing=ColumnTransformer(transformers=
                                    make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
                                    ).set_output(transform="pandas")