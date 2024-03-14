import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations

datasets = ['eicu', 'mimic', 'miiv', 'hirid']

baseline_alphas = pd.read_parquet('baseline_results/Lasso/group_alphas_forward_selection.parquet')

for r in range(2,4):
    #baseline_paths = [f'lasso_train_on_{group_combination}_forward_selection_results.parquet' for group_combination in group_combs]
    
    if r == 2:
        fig, axs = plt.subplots(2,3,figsize=(12,9))
    if r == 3:
        fig, axs = plt.subplots(2,2,figsize=(12,9))
    i = 0
    for group_combination in combinations(datasets,r):
        alpha = baseline_alphas[baseline_alphas['group name'].apply(lambda x: len(x) == len(group_combination) and all(elem in x for elem in group_combination))]['alpha'].iloc[0]
        if r == 2:
            row = i%2
            col = i%3
        if r == 3:
            row, col = divmod(i,2)
        ax = axs[row,col]
        data = pd.read_parquet(f'baseline_results/Lasso/lasso_train_on_{group_combination}_forward_selection_results.parquet')
        for col in data.columns:
            splitted_col = col.split(' ')
            if 'name' not in col:
                if 'train' in col:
                    ax.plot(range(0,51), data[col], 'ko-', label=f'train, alpha={round(alpha,4)}', ms=2)
                if 'test' in col:
                    if 'eicu' in col: 
                        ax.plot(range(0,51), data[col], 'bo-', label=splitted_col[2], ms=2)
                    if 'mimic' in col: 
                        ax.plot(range(0,51), data[col], 'ro-', label=splitted_col[2], ms=2)
                    if 'miiv' in col: 
                        ax.plot(range(0,51), data[col], 'mo-', label=splitted_col[2], ms=2)
                    if 'hirid' in col: 
                        ax.plot(range(0,51), data[col], 'co-', label=splitted_col[2], ms=2)
        ax.legend()
        ax.grid()
        ax.set_xlabel('Number of features')
        ax.set_title(f'Trained on {group_combination}')
        i += 1
    fig.suptitle('Lasso Baseline')
plt.tight_layout()

for r in range(2,4):
    if r == 2:
        fig, axs = plt.subplots(2,3,figsize=(12,9))
    if r == 3:
        fig, axs = plt.subplots(2,2,figsize=(12,9))
    i = 0
    for group_combination in combinations(datasets,r):
        if r == 2:
            row = i%2
            col = i%3
        if r == 3:
            row, col = divmod(i,2)
        ax = axs[row,col]
        data = pd.read_parquet(f'dsl_results/multiple alpha/multiple_alphas_train_on_{group_combination}_forward_selection_results.parquet')
        alphas = data['alpha'].unique()
        for alpha in alphas: 
            for col in data.columns:
                splitted_col = col.split(' ')
                if 'name' not in col:
                    if 'train' in col:
                        ax.plot(range(0,51), data[data['alpha']==alpha][col].iloc[0], 'ko-', ms=2, alpha=0.3)
                    if 'test' in col:
                        if 'eicu' in col: 
                            ax.plot(range(0,51), data[data['alpha']==alpha][col].iloc[0], 'bo-', ms=2)
                        if 'mimic' in col: 
                            ax.plot(range(0,51), data[data['alpha']==alpha][col].iloc[0], 'ro-', ms=2)
                        if 'miiv' in col: 
                            ax.plot(range(0,51), data[data['alpha']==alpha][col].iloc[0], 'mo-', ms=2)
                        if 'hirid' in col: 
                            ax.plot(range(0,51), data[data['alpha']==alpha][col].iloc[0], 'co-', ms=2)
        ax.grid()
        ax.set_xlabel('Number of features')
        ax.set_title(f'Trained on {group_combination}')
        i += 1
    fig.suptitle('DSL')
    plt.tight_layout()
plt.show()