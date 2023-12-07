from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from lightgbm import LGBMRegressor
from ivmodels import AnchorRegression

model = 'ols'
outcome = 'hr'
n_seeds = 10

sources = ['eicu', 'hirid', 'mimic', 'miiv']

training_source = 'eicu'

anchor_columns = ['hospital_id']

grid_search = True
evaluation_on_target = False

methods = [
    'ols',
    'ridge',
    'lgbm',
    'rf',
    'anchor'
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
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]
    }  
}

n_fine_tuning = [0, 25, 50, 100, 200, 400, 800, 1600]

Regressor = {
    'ols': LinearRegression(),
    'ridge': Ridge(),
    'lgbm': LGBMRegressor(),
    'rf': LGBMRegressor(),
    'anchor': AnchorRegression()
}

Preprocessing = {
    'ols': ColumnTransformer(
        transformers=make_feature_preprocessing(missing_indicator=True)
        ).set_output(transform="pandas"),
    'ridge': ColumnTransformer(
        transformers=make_feature_preprocessing(missing_indicator=True))
        .set_output(transform="pandas"),
    'lgbm': ColumnTransformer(
        transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
        ).set_output(transform="pandas"),
    'rf': ColumnTransformer(
        transformers=make_feature_preprocessing(missing_indicator=False, categorical_indicator=False)
        ).set_output(transform="pandas"),
    'anchor': ColumnTransformer(
        make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
        ).set_output(transform="pandas")
}