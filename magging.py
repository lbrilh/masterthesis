from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.linalg import eigvals
from data import load_data, save_data, results_exist
import pandas as pd

_data = load_data(outcome='hr')

ridge_hyper={
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
}

Regressor=Ridge()
Preprocessing=ColumnTransformer(transformers=
                                make_feature_preprocessing(missing_indicator=True)
                                ).set_output(transform="pandas")

pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessing),
    ('model', Regressor)
])


if not results_exist('parameters_ridge.pkl'):
    params = []
    for source in ['eicu','hirid','mimic','miiv']:
        search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in ridge_hyper.items()})
        search.fit(_data[source]['train'], _data[source]['train']['outcome'])
        best_model = search.best_estimator_
        model_params = best_model.named_steps['model'].coef_
        params.append({
            f'Parameters on {source}': model_params
        })
        print(f'Completed on {source}')

    save_data(path='parameters_ridge.pkl',results=params)
_params_data=load_data('parameters_ridge.pkl')

params = pd.concat([_params_data[0]['Parameters on eicu'].
                    _params_data[1]['Parameters on hirid'],
                    _params_data[2]['Parameters on mimic'],
                    _params_data[3]['Parameters on miiv']])

X_eicu=Preprocessing.fit_transform(_data['eicu']['train'])
X_hirid=Preprocessing.fit_transform(_data['hirid']['train'])
X_mimic=Preprocessing.fit_transform(_data['mimic']['train'])
X_miiv=Preprocessing.fit_transform(_data['miiv']['train'])
result = pd.concat([X_eicu, X_hirid, X_mimic, X_miiv], axis=0)

eigenvalues = eigvals(result)

print(eigenvalues)