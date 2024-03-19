'''
    This script visualizes the performance of Lasso and DSL (Data Shared Lasso) models trained on combinations of
    four different healthcare datasets (`eicu`, `mimic`, `miiv`, `hirid`). It creates plots to compare the models based on
    the number of features selected through forward selection and their performance metrics. The script is structured into
    three main blocks, each targeting a specific aspect of the visualization.
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from itertools import combinations

outcome = 'map'
datasets = ['eicu', 'mimic', 'miiv', 'hirid']

# This block generates plots for the Lasso baseline model's performance metrics, such as train and test scores, 
# for each combination of 2 and 3 datasets.
baseline_alphas = pd.read_parquet(f'baseline_results/lasso/{outcome}/{outcome}_group_alphas_forward_selection.parquet')
for r in range(2,4):    
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
        data = pd.read_parquet(f'baseline_results/lasso/{outcome}/{outcome}_lasso_train_on_{group_combination}_forward_selection_results.parquet')
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

# This block visualizes the DSL model's performance for each combination of 2 and 3 datasets, considering multiple alpha values.
# It highlights how the performance varies with different regularization strengths by mapping alpha values to a color gradient.
for r in range(2, 4):
    if r == 2:
        fig, axs = plt.subplots(2, 3, figsize=(12, 9))
    if r == 3:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    i = 0
    for group_combination in combinations(datasets, r):
        if r == 2:
            row = i % 2
            col = i % 3
        if r == 3:
            row, col = divmod(i, 2)
        ax = axs[row, col]
        data = pd.read_parquet(
            f'dsl_results/multiple alpha/{outcome}/{outcome}_multiple_alphas_train_on_{group_combination}_forward_selection_results.parquet')
        alphas = data['alpha'].unique()
        colors = {'eicu': 'blue', 'mimic': 'red', 'miiv': 'magenta', 'hirid': 'cyan'}
        norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
        cmap = plt.get_cmap('viridis')
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        for alpha in alphas:
            for col in data.columns:
                splitted_col = col.split(' ')
                dataset_name = None
                for dataset in datasets:
                    if dataset in splitted_col:
                        dataset_name = dataset
                        break
                if dataset_name:
                    if 'name' not in col:
                        # if 'train' in col:
                        #     color = scalar_map.to_rgba(alpha)
                        #     ax.plot(range(0, 51), data[data['alpha'] == alpha][col].iloc[0],
                        #             color=color, alpha=1, marker='o', linestyle='-',
                        #             ms=2, label='train')
                        if 'test' in col:
                            color = scalar_map.to_rgba(alpha)
                            ax.plot(range(0, 51), data[data['alpha'] == alpha][col].iloc[0],
                                    color=color, alpha=0.8, marker='o', linestyle='-',
                                    ms=2, label='test')
        ax.grid()
        ax.set_xlabel('Number of features')
        ax.set_title(f'Trained on {group_combination}')
        i += 1
    fig.suptitle('DSL')
    plt.tight_layout()
    plt.colorbar(scalar_map, ax=axs, orientation='vertical', label='Alpha')

# Adds a dedicated color legend for better dataset distinction in DSL model performance visualization.
colors = {'eicu': 'blue', 'mimic': 'red', 'miiv': 'magenta', 'hirid': 'cyan'}
for r in range(2, 4):
    if r == 2:
        fig, axs = plt.subplots(2,3,figsize=(12,9))
        fig.subplots_adjust(right=0.8)
    if r == 3:
        fig, axs = plt.subplots(2,2,figsize=(12,9))
        fig.subplots_adjust(right=0.75)
    i = 0
    for group_combination in combinations(datasets, r):
        if r == 2:
            row = i%2
            col = i%3
        if r == 3:
            row, col = divmod(i,2)
        ax = axs[row, col]
        data = pd.read_parquet(f'dsl_results/multiple alpha/{outcome}/{outcome}_multiple_alphas_train_on_{group_combination}_forward_selection_results.parquet')
        alphas = data['alpha'].unique()
        for alpha in alphas:
            for col in data.columns:
                splitted_col = col.split(' ')
                if 'name' not in col:
                    # if 'train' in col: # ctrl + /
                    #     ax.plot(range(0, 51), data[data['alpha']==alpha][col].iloc[0], 'ko-', ms=2, alpha=0.3)
                    if 'test' in col:
                        for dataset_name, color in colors.items():
                            if dataset_name in col:
                                ax.plot(range(0, 51), data[data['alpha']==alpha][col].iloc[0], color=color, marker='o', linestyle='-', ms=2, alpha=0.8)
        ax.grid()
        ax.set_xlabel('Number of features')
        ax.set_title(f'Trained on {group_combination}')
        i += 1
    # Create a color legend subplot
    ax_legend = fig.add_subplot(1, 1, 1, frameon=False)
    ax_legend.axis('off')
    for dataset_name, color in colors.items():
        ax_legend.plot([], [], color=color, label=dataset_name)
    ax_legend.legend(title='Color Legend', loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig.suptitle('DSL')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()