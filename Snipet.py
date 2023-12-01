outcome = "hr"

sources = ['eicu', 'hirid', 'mimic', 'miiv']
regressors = ['lgbm', 'rf', 'ols', 'anchor']
_data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)
pipelines = {'lgbm': pipeline_lgbm,
             'rf': pipeline_rf,
             'ols': pipeline_ols,
             'anchor': pipeline_anchor}
params = {}
params['lgbm'] = {
    'boosting_type': ['gbdt'],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 800],
    'num_leaves': [50, 200, 1024],
    'feature_fraction': [0.5, 0.9],
    'verbose': [-1]
}
params['rf'] = {
    'boosting_type': ['gbdt'],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 800],
    'num_leaves': [50, 200, 1024],
    'feature_fraction': [0.5, 0.9],
    'verbose': [-1]
}
params['anchor'] = {
    'gamma': [1, 10, 10000],
    'instrument_regex': ['anchor'],
    'alpha': [0.00001, 0.001, 0.1]
}

mse_grid_search = {}

for name, pipe in pipelines.items():
    print(name)
    if name not in ['ols', 'anchor']:
        search = GridSearchCV(pipe, param_grid= {'model__' + key : value for key, value in params[name].items()})
        search.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])
        print('finsihed GCV')
        pipe.set_params(**search.best_params_)
    
    pipe.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])
    
    for source in sources: 
        print(source)
        if source != 'eicu':
            if name not in ['ols', 'anchor']:
                mse_grid_search[name] = {'parameters': search.best_params_,
                'MSE on {source}' : mean_squared_error(_data[source]['train']['outcome'], pipe.predict(_data[source]['train']))}
            else: 
                mse_grid_search[name] = {'parameters': None,
                'MSE on {source}' : mean_squared_error(_data[source]['train']['outcome'], pipe.predict(_data[source]['train']))}
        print(f'Completed {name} run on {source}')


results = {}

for name, pipe in pipelines.items():
    if name not in ['ols']:
        results[name] = {}
        print(name)
        for comb, param_set in enumerate(itertools.product(*params[name].values())):
            para = dict(zip(params[name].keys(), param_set))
            pipe.named_steps['model'].set_params(**para)
            pipe.fit(_data['eicu']['train'], _data['eicu']['train']['outcome'])
            results[name][comb] = {}
            for source in sources: 
                results[name][comb][source] = {}
                if source != 'eicu':
                    for n in [25, 50, 100, 200, 400, 800, 1600]:
                        y_pred_eval = pipe.predict(_data[source]['test'].head(n))
                        y_pred_test = pipe.predict(_data[source]['train'])
                        
                        mse_eval = mean_squared_error(_data[source]['test']['outcome'].head(n), y_pred_eval)
                        mse_test = mean_squared_error(_data[source]['train']['outcome'], y_pred_test)
                        
                        results[name][comb][source][n] = {
                            'params': para,
                            'MSE on Eval. Set from Target': mse_eval,
                            'MSE on Target': mse_test
                        }
                print(f'finished {comb} on source {source}')
        clear_output()