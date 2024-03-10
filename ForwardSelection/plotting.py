import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations

datasets = ['eicu', 'mimic', 'miiv', 'hirid']
baseline_paths = [f'{dataset}_forward_selection_results.parquet' for dataset in datasets]

fig, axs = plt.subplots(2,2,figsize=(12,9))
i = 0
for path in baseline_paths:
    splitted_path = path.split('_')
    row, col = divmod(i,2)
    ax = axs[row,col]
    data = pd.read_parquet(f'baseline_results/{path}')
    for col in data.columns:
        splitted_col = col.split(' ')
        if 'name' not in col:
            if 'train' in col:
                ax.plot(range(0,51), data[col], 'ko-', label=col, ms=2)
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
    ax.set_title(f'Trained on {splitted_path[0]}')
    i += 1
fig.suptitle('OLS Baseline')
plt.tight_layout()

for r in range(2,4):
    dsl_paths = [f'train_on_{group_combination}_forward_selection_results.parquet' for group_combination in combinations(datasets, r)]
    if r == 2:
        fig, axs = plt.subplots(2,3,figsize=(12,9))
    if r == 3:
        fig, axs = plt.subplots(2,2,figsize=(12,9))
    i = 0
    for path in dsl_paths:
        splitted_path = path.split('_')
        if r == 2:
            row = i%2
            col = i%3
        if r == 3:
            row, col = divmod(i,2)
        ax = axs[row,col]
        data = pd.read_parquet(f'dsl_results/{path}')
        for col in data.columns:
            splitted_col = col.split(' ')
            if 'name' not in col:
                if 'train' in col:
                    ax.plot(range(0,51), data[col], 'ko-', label=col, ms=2)
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
        ax.set_title(f'Trained on {splitted_path[2]}')
        i += 1
    fig.suptitle('DSL')
    plt.tight_layout()


plt.show()