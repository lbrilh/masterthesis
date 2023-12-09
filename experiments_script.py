from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import itertools
from numpy.linalg import eig, det
from plotting import plotting
from set_up import grid_search, evaluation_on_target, hyper_parameters, model, outcome, n_seeds, sources, training_source, anchor_columns, methods, Regressor, Preprocessing
from data import load_data, results_exist, save_data

assert model in methods


_data = load_data(outcome)


if model == 'lgbm' or model == 'rf':
    sex_index = _data[training_source]['train'].columns.get_loc('sex')
    _data[training_source]['train']['sex'] = _data[training_source]['train']['sex'].astype('category')


pipeline = Pipeline(steps=[
    ('preprocessing', Preprocessing),
    ('model', Regressor)
])


if grid_search: 
    mse_grid_search = {}
    if not results_exist(model,grid=grid_search):
        if model in ['ols']:
            print(f'No hyperparameters for {model}. Skip GridCV')
        else:
            print(f'Start with GridCV for {model}: \n')
            search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_parameters[model].items()})
            search.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
            print('finsihed GCV')
            pipeline.set_params(**search.best_params_)
        pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
        print(f'Start evaluation with parameter selection from grid search for {model}:')
        for source in sources: 
            if source != training_source:
                mse = mean_squared_error(_data[source]['train']['outcome'], pipeline.predict(_data[source]['train']))
                if model not in ['ols']:
                    mse_grid_search[model] = {
                        'parameters': search.best_params_,
                        'MSE on {source}': mse
                        }
                else: 
                    mse_grid_search[model] = {
                        'parameters': None,
                        'MSE on {source}': mse
                        }
            print(f'Completed {model} run on {source}')
        save_data(model, mse_grid_search, grid=grid_search)


if evaluation_on_target:
    path_results_model = model + '_results.pkl'
    if not results_exist(model, eval=evaluation_on_target)
        sample_seeds = list(range(n_seeds))
        results = []
        if model != 'ols':
            print(f"Hyperparametrs for {model} will be chosen via performance on fine-tuning set from target \n")
            hyper_para_combinations = list(itertools.product(*hyper_parameters[model].values()))
            num_combinations = len(hyper_para_combinations)
            for comb, hyper_para_set in enumerate(itertools.product(*hyper_parameters[model].values())):
                hyper_para = dict(zip(hyper_parameters[model].keys(), hyper_para_set))            
                pipeline.named_steps['model'].set_params(**hyper_para)
                pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
                for source in sources: 
                    if source != training_source:
                        for sample_seed in sample_seeds:      
                            Xy_target_tuning = _data[source]["test"].sample(
                            frac=1, random_state=sample_seed
                            )
                            Xy_target_evaluation = _data[source]['train']
                            for n in n_fine_tuning:
                                y_pred_tuning = pipeline.predict(Xy_target_tuning[:n])
                                y_pred_evaluation = pipeline.predict(Xy_target_evaluation)
                                mse_tuning = mean_squared_error(Xy_target_tuning['outcome'][:n], y_pred_tuning)
                                mse_evaluation = mean_squared_error(Xy_target_evaluation['outcome'], y_pred_evaluation)
                                results.append({
                                        'comb_nr': comb,
                                        'parameters': hyper_para, 
                                        "target": source,
                                        "n_test": n,
                                        "sample_seed": sample_seed,
                                        'mse tuning': mse_tuning,
                                        'mse target': mse_evaluation
                                    })
                            print(f'finished combination {comb+1} from {num_combinations} with sample {sample_seed} on source {source} using {model}')
        elif model == 'ols':
            pipeline.fit(_data[training_source]['train'], _data[training_source]['train']['outcome'])
            sing = min(pipeline.named_steps['model'].singular_)
            results.append({'smalles singular value of X': sing})
            for source in sources: 
                Xy_target_evaluation = _data[source]['train']
                y_pred_evaluation = pipeline.predict(Xy_target_evaluation)
                mse_evaluation = mean_squared_error(Xy_target_evaluation['outcome'], y_pred_evaluation)
                results.append({
                    "target": source,
                    'mse target': mse_evaluation
                })
                print(f'finished on source {source} using {model}\n')
        save_data(model, results, eval=evaluation_on_target)

plotting(model=model, methods=methods, sources=sources, training_source=training_source)

print('Script completed with no erros\n')