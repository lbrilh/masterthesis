import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("mse_robust.parquet")
print(df)

fig, axs = plt.subplots(3,1,figsize=(12,9))

for nr_groups in range(0,3):
    ordered_columns = ['intercept']
    ordered_columns += [f'{i} feat.' for i in range(1, 1+51*(2+nr_groups))]
    ax = axs[nr_groups]
    df_nr_groups = df[f"Nr Groups {nr_groups+1}"]
    for target, name in enumerate(['eicu', 'mimic', 'miiv', 'hirid']):
        target_df = pd.DataFrame(df[f'Nr Groups {nr_groups+1}'].iloc[target])
        ax.plot(range(0, 1 + 51*(2+nr_groups)), target_df[ordered_columns].to_numpy().reshape(-1), 'o-', label=name, alpha=0.5, ms=3)
        ax.set_xlim(left=-1)
        ax.legend()
plt.tight_layout()
plt.show()