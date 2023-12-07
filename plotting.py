import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os 
from set_up import sources, methods, training_source

#def plot_tuning(train_source: str, sources: list[str], methods: list[str]):
pattern = r'*_results.pkl'
# Retrieve a list of file paths that match the pattern
file_paths = glob.glob(pattern)
print(file_paths)

for source in sources:
    plt.figure(figsize=(10, 6))
    plt.title(f"MSE vs Number of tuning Data Points on {source} with parameter training on {training_source}")
    plt.xlabel("Number of tuning Points")
    plt.ylabel("MSE")
    for model in methods: 
        with open(f'{model}_results.pkl', 'rb') as data:
            _data=pickle.load(data)
        df = pd.DataFrame(_data)
        results = df[df['target']==source].groupby(by=['n_test','comb_nr'])[['mse evaluation', 'mse target']].mean().sort_index().reset_index()
        min_mse_eval_indices = results.groupby('n_test')['mse evaluation'].idxmin()
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