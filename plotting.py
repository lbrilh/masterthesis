from data import retrieve_paths, load_data_plotting
import pandas as pd
import matplotlib.pyplot as plt

def plotting(model: str, methods: list[str], sources: list[str], training_source: str, pattern=r'*_results.pkl') -> None:
    file_paths=retrieve_paths(pattern)
    print(file_paths)

    for source in sources:
        if source==training_source:
            print('Lorem Ipsum')
        else: 
            plt.figure(figsize=(10, 6))
            plt.title(f"MSE vs Number of tuning Data Points on {source} with parameter training on {training_source}")
            plt.xlabel("Number of tuning Points")
            plt.ylabel("MSE")
            for model in methods: 
                _data=load_data_plotting(model)
                df = pd.DataFrame(_data)
                results = df[df['target']==source].groupby(by=['n_test','comb_nr'])[['mse tuning', 'mse target']].mean().sort_index().reset_index()
                min_mse_eval_indices = results.groupby('n_test')['mse tuning'].idxmin()
                mse_targets = results.loc[min_mse_eval_indices][['n_test', 'comb_nr', 'mse target']]
                if model=='anchor': 
                    unique_df = df.drop_duplicates(subset=['comb_nr'])  
                    for combination in mse_targets['comb_nr']:
                        print(unique_df[unique_df['comb_nr']==combination][['parameters']])
                plt.plot(mse_targets['n_test'], mse_targets['mse target'], '-o', label=model, linewidth=2)
            plt.legend()
    plt.show()
    print('Script successfully executed')


def plot_tuning_by_gamma(sources: list[str], training_source: str, n_tuning_points: list[int]) -> None:
    _data=load_data_plotting(tuning_by_gamma)
    _df=pd.DataFrame(_data)
    _df.drop(columns=['parameters','sample_seed','comb_nr', 'mse tuning'], inplace=True)    
    for source in sources:
        if not source==training_sourceS: 
            df = _df
            df.drop(columns=['target'],inplace=True)
            df=df.groupby(by=['n_test','gamma']).mean().reset_index()
            plt.figure(figsize=(10, 6))
            plt.title(f"MSE vs gamma on {source} with parameter training on {training_source}")
            plt.xlabel(f"\gamma")
            plt.ylabel("MSE")
            for n in n_tuning_points:
                df_=df[df['n_test']==n]
                plt.plot(df_['gamma'],df_['mse target'], '-o', linewidth=2)
            plt.legend()
            plt.show()
    print('Script successfully executed')