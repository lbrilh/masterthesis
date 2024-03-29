''' 
    Plot the minimal mean test mse conditioned on the number of fine-tuning data points for each estimator.
    Additionally, we plot the Mean Squared Error (MSE) vs gamma for tuning data points across different sources for Anchor Regression.
'''

import os
import glob

import pandas as pd
import matplotlib.pyplot as plt

# Function to retrieve all paths
def retrieve_paths(pattern):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'parquet')
    file_paths = glob.glob(os.path.join(file_path, pattern))
    return file_paths

# Generate plots depicting minimal Mean Squared Error (MSE) vs Number of tuning Data Points for different sources
def plotting(sources, training_source, pattern=r'*_results.parquet'):
    file_paths=retrieve_paths(pattern)
    for source in sources:
        if source==training_source:
            print('Skip train source in plotting')
        else: 
            plt.figure(figsize=(10, 10))
            #plt.title(f"MSE vs Number of tuning Data Points on {source} with parameter training on {training_source}")
            plt.xlabel("Number of fine-tuning points", fontsize=16)
            plt.ylabel("MSE", fontsize=16)
            for file in file_paths: 
                _data=load_data_plotting(path=file)
                df = pd.DataFrame(_data)
                if 'tuning' not in file:
                    if 'grid' in file:
                        if 'ols' not in file: 
                            if 'rf' in file:
                                df=df[lambda x: x['target'].eq(source)]
                                plt.axhline(y=df['MSE'].iloc[0], linestyle = '--', color='purple', label='RF Grid', linewidth=3) 
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                            if 'lgbm' in file:
                                df=df[lambda x: x['target'].eq(source)]
                                plt.axhline(y=df['MSE'].iloc[0], linestyle = '--', color='red', label='GBDT Grid', linewidth=3) 
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                            if 'ridge' in file:
                                df=df[lambda x: x['target'].eq(source)]
                                plt.axhline(y=df['MSE'].iloc[0], linestyle = '--', color='brown', label='Ridge Grid', linewidth=3)
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600]) 
                    else: 
                        group_columns = ["target", "n_test", "sample_seed"]
                        if 'refit' in file:
                            metric='mse target'
                        else: 
                            metric='mse tuning'
                        print(df)
                        df[f"{metric}_min"] = df.groupby(by=group_columns)[metric].transform("min")
                        df = df[lambda x: x[metric].eq(x[f"{metric}_min"])]
                        df = df.drop(columns=f"{metric}_min")
                        df.sort_values(group_columns,inplace=True)
                        df=df[lambda x: x['target'].eq(source)].groupby('n_test')['mse target'].mean().sort_index().reset_index()
                        if 'rf' in file:
                                plt.plot(df['n_test'], df['mse target'], '-o', label='RF', linewidth=3)
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                        if 'lgbm' in file:
                            if 'refit' in file: 
                                plt.plot(df['n_test'], df['mse target'], '-o', label='RefitLGBM', linewidth=3)
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                            elif 'anchor' not in file:
                                plt.plot(df['n_test'], df['mse target'], '-o', label='GBDT', linewidth=3)
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                        if 'ridge' in file:
                            plt.plot(df['n_test'], df['mse target'], '-o', label='Ridge', linewidth=3)
                            plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                        if 'anchor' in file:
                            if 'lgbm' in file: 
                                plt.plot(df['n_test'], df['mse target'], '-o', label='AnchorBoost', linewidth=3)
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                            else:
                                plt.plot(df['n_test'], df['mse target'], '-o', label='Anchor', linewidth=3)
                                plt.xticks(ticks=[25, 50, 100, 200, 400, 800, 1600])
                    plt.xscale('log')
                plt.xticks(fontsize=15, size=15)
                plt.tick_params(axis='x', bottom=False, which='both',  labelbottom=False)
                plt.yticks(fontsize=15, size=15)
                plt.tick_params(axis='y', width=0)
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                if source != 'mimic':
                    plt.legend(fontsize=16)    
            plt.savefig(f'plots/fine_tuning/{training_source} on {source}')
    plt.show()
    print('Script successfully executed')


def load_data_plotting(model=None,path=None):
    if path:
        file_path=path
    else: 
        current_directory = os.getcwd()
        relative_path = os.path.join('parquet', f'{model}_results.parquet')
        file_path = os.path.join(current_directory, relative_path)
    _data=pd.read_parquet(file_path, engine='pyarrow')
    return _data

# Plot the Mean Squared Error (MSE) vs gamma for tuning data points across different sources.
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
                plt.plot(df['gamma'].unique(),df['mse target'], '-o', linewidth=3, label=f'n_test={n}')
                plt.xscale('log')
            plt.legend()
            print('New source \n')
            plt.savefig(f'plots/tuning by gamma/tuning_by_gamma_on_{source}')
    plt.show()
print('Script successfully executed')