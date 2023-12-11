from data import retrieve_paths, load_data_plotting
import pandas as pd
import matplotlib.pyplot as plt

def plotting(model: str, methods: list[str], sources: list[str], training_source: str, pattern=r'*_results.pkl'):
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
            #for model in methods: 
            _data=load_data_plotting(model)
            df = pd.DataFrame(_data)
            group_columns = ["target", "n_test", "sample_seed"]
            metric='mse tuning' #could be also mse target
            df[f"{metric}_min"] = df.groupby(group_columns)[metric].transform("min")
            df = df[lambda x: x[metric].eq(x[f"{metric}_min"])]
            df = df.drop(columns=f"{metric}_min")
            df.sort_values(group_columns,inplace=True)
            df=df[lambda x: x['target'].eq(source)].groupby('n_test')['mse target'].mean().sort_index().reset_index()
            plt.plot(df['n_test'], df['mse target'], '-o', label=model, linewidth=2)
            plt.legend()
    plt.show()
    print('Script successfully executed')


def plot_tuning_by_gamma(sources: list[str], training_source: str, n_tuning_points: list[int]):
    _data=load_data_plotting('tuning_by_gamma')
    _df=pd.DataFrame(_data)
    for source in sources:
        if not source==training_source: 
            results = _df[_df['target']==source].groupby(by=['n_test','gamma'])[['mse tuning', 'mse target']].mean().sort_index().reset_index()
            plt.figure(figsize=(10, 6))
            plt.title(f"MSE vs gamma on {source} with parameter training on {training_source}")
            plt.xlabel("gamma")
            plt.ylabel("MSE")
            for n in n_tuning_points:
                df=results[results['n_test']==n]
                plt.plot(df['gamma'],df['mse target'], '-o', linewidth=2, label=f'n_test={n}')
            plt.legend()
            plt.show()
    print('Script successfully executed')