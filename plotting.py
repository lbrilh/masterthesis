import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os 

def plotting(model: str, methods: list[str], sources: list[str], training_source: str, pattern=r'*_results.pkl') -> None:

    # Retrieve a list of file paths that match the pattern
    file_paths = glob.glob(pattern)
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
                with open(f'{model}_results.pkl', 'rb') as data:
                    _data=pickle.load(data)
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
    plt.savefig("Anchor")
    print('Script successfully executed')

def plot_tuning_by_gamma(model: str, gammas: list[int], sources: list[str], training_source: str) -> None:
    with open('tuning_by_gamma_results.pkl','rb') as data:
        _data=pickle.load(data)

    df=pd.DataFrame(_data)
    print(df.head())
    ########### Sort it by n_test_tuning and plot for different gammas (x-axis)
    for source in sources:
        if source==training_source:
            print('Lorem Ipsum')
        else: 
            plt.figure(figsize=(10, 6))
            plt.title(f"MSE vs Number of tuning Data Points on {source} with parameter training on {training_source}")
            plt.xlabel("Number of tuning Points")
            plt.ylabel("MSE")
            for model in methods: 
                with open(f'{model}_results.pkl', 'rb') as data:
                    _data=pickle.load(data)
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
    plt.savefig("Anchor")
    print('Script successfully executed')