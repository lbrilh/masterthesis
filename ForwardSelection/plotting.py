import pandas as pd
import matplotlib.pyplot as plt

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
plt.show()