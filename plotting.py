from data import retrieve_paths, load_data_plotting
import pandas as pd
import matplotlib.pyplot as plt

def plotting(methods, sources, training_source, pattern=r'*_results.pkl'):
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
            for file in file_paths: 
                print(file)
                _data=load_data_plotting(path=file)
                df = pd.DataFrame(_data)
                if 'grid' in file or 'tuning' in file:
                    print('lorem ipsum need to be implemented')
                    #print(df.head())
                    #plt.axvline(x=df['MSE on {source}'], linestyle='--', label=file)
                else: 
                    group_columns = ["target", "n_test", "sample_seed"]
                    metric='mse tuning'
                    df[f"{metric}_min"] = df.groupby(group_columns)[metric].transform("min")
                    df = df[lambda x: x[metric].eq(x[f"{metric}_min"])]
                    df = df.drop(columns=f"{metric}_min")
                    df.sort_values(group_columns,inplace=True)
                    df=df[lambda x: x['target'].eq(source)].groupby('n_test')['mse target'].mean().sort_index().reset_index()
                    plt.plot(df['n_test'], df['mse target'], '-o', label=file, linewidth=2)
                    plt.xscale('log')
                plt.legend()
    plt.show()
    print('Script successfully executed')


def plot_tuning_by_gamma(sources, training_source, n_tuning_points):
    _data=load_data_plotting('tuning_by_gamma')
    _df=pd.DataFrame(_data)
    group_columns = ["target", "n_test", "sample_seed", "gamma"]
    metric='mse tuning' #could be also mse target
    _df[f"{metric}_min"] = _df.groupby(group_columns)[metric].transform("min")
    _df = _df[lambda x: x[metric].eq(x[f"{metric}_min"])]
    _df = _df.drop(columns=f"{metric}_min")
    _df.sort_values(group_columns,inplace=True)
    for source in sources:
        if not source==training_source: 
            print(f'Target: {source}')
            plt.figure(figsize=(10, 6))
            plt.title(f"MSE vs gamma on {source} with parameter training on {training_source}")
            plt.xlabel("gamma")
            plt.ylabel("MSE")
            for n in n_tuning_points:
                df=_df[lambda x: x['target'].eq(source) & x['n_test'].eq(n)].groupby(by=['gamma'])[['mse target']].mean().reset_index()
                print(df)
                plt.plot(df['gamma'].unique(),df['mse target'], '-o', linewidth=2, label=f'n_test={n}')
                plt.xscale('log')
            plt.legend()
            print('New source \n')
    plt.show()
    print('Script successfully executed')