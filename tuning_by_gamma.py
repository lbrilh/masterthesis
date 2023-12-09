import itertools
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from ivmodels import AnchorRegression
from icu_experiments.load_data import load_data_for_prediction
from icu_experiments.preprocessing import make_feature_preprocessing, make_anchor_preprocessing
from plotting import plot_tuning_by_gamma
from data import load_data, save_data
from set_up import outcome, n_seeds, sources, training_source, anchor_columns, n_fine_tuning, overwrite, model, hyperparameters


if overwrite: 
    sample_seeds = list(range(n_seeds))
    results = []
    # Check if data has already been processed 
    _data=load_data(outcome)


    print(f"Hyperparametrs will be chosen via performance on fine-tuning set from target \n")
    hyper_para_combinations = list(itertools.product(*hyper_parameters.values()))
    num_combinations = len(hyper_para_combinations)
    for gamma in gammas: 
        ################ set gamma correclty 
        ################ Adjust set_up.py for this case 
        '''
        gammas = [1, 3.16, 10, 31.6, 100, 316, 1000, 3162, 10000]

hyper_parameters ={
    'instrument_regex': ['anchor'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'l1_ratio': [0, 0.2, 0.5, 0.8, 1]
}  
        '''
        Regressor=AnchorRegression()
        Preprocessing=ColumnTransformer(transformers=
                                        make_anchor_preprocessing(anchor_columns) + make_feature_preprocessing(missing_indicator=True)
                                        ).set_output(transform="pandas")
        pipeline = Pipeline(steps=[
        ('preprocessing', Preprocessing),
        ('model', Regressor)
        ])

        for comb, hyper_para_set in enumerate(itertools.product(*hyper_parameters.values())):
            hyper_para = dict(zip(hyper_parameters.keys(), hyper_para_set))            
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
                                'gamma': gamma, 
                                'comb_nr': comb,
                                'parameters': hyper_para, 
                                "target": source,
                                "n_test": n,
                                "sample_seed": sample_seed,
                                'mse tuning': mse_tuning,
                                'mse target': mse_evaluation
                            })
                        print(f'finished combination {comb+1} from {num_combinations} with sample {sample_seed} on source {source}')


    save_data(path='tuning_by_gamma_results.pkl',results=results)

plot_tuning_by_gamma(sources=sources, training_source=training_source, n_tuning_points=n_fine_tuning)