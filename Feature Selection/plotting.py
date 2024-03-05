# TODO: Make plots a bit nicer (i.e. same groups have same line color); compare with magging and prior results; which individual groups are chosen?
import pandas as pd
import matplotlib.pyplot as plt
import itertools

df = pd.read_parquet("mse_robust.parquet")
df_features = pd.read_parquet('dsl_feature_list.parquet')
datasets = ['eicu', 'mimic', 'miiv', 'hirid']

for r in range(1,4):
    fig, axs = plt.subplots(2,2, figsize=(12,9))
    for i, target in enumerate(datasets):
        row, col = divmod(i,2)
        ax = axs[row, col]
        ordered_columns = ['intercept']
        ordered_columns += [f'{i} feat.' for i in range(1, 1+51*(1+r))]
        target_df = pd.DataFrame(df[f'Nr Groups {r}'].iloc[i])
        feature_names = pd.DataFrame(df_features[f'Nr Groups {r}'].iloc[i])
        feature_names.dropna(axis=1, inplace=True)
        target_df.dropna(inplace=True, axis=1)
        for combination in target_df.columns: 
            combination_feature_names = pd.DataFrame({'features' :(feature_names[combination].to_list())[0]})
            colors = ['purple']
            colors += ['k' if 'passthrough' in combination_feature_names.iloc[i].values[0]
                       else 'r' for i in range(combination_feature_names.shape[0])]
            results = [float(arr[0]) for arr in target_df[combination][ordered_columns].to_numpy()]
            ax.plot(range(0, 1 + 51*(1+r)), results, '-', label=combination)
            ax.legend()
            ax.scatter(range(0, 1 + 51*(1+r)), results, c=colors, alpha=0.7)
            ax.set_title(f'target: {target}')
            ax.grid(True)
        if r == 1:
            if target == 'eicu':
                ax.set_ylim((147.8,153))
            if target == 'hirid':
                ax.set_ylim((137,153))
            if target == 'miiv':
                ax.set_ylim((129,250))
            if target == 'mimic':
                ax.set_ylim((145,275))
        if r == 2:
            if target == 'eicu':
                ax.set_ylim((148,152))
            if target == 'hirid':
                ax.set_ylim((130,155))
            if target == 'miiv':
                ax.set_ylim((129,250))
            if target == 'mimic':
                ax.set_ylim((145,275))
        if r == 3:
            if target == 'hirid':
                ax.set_ylim((151, 153))
            else:
                ax.set_ylim((147.5,155))
        #ax.set_yticklabels(ax.get_yticks().astype(int))
        plt.tight_layout()
plt.show()