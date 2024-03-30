'''
    Generate bar plots for the coefficients of Lasso fitted on each dataset separately.
    The response variable is "outcome".
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outcome = 'hr'
datasets = ['eicu', 'mimic', 'miiv', 'hirid']

fig, axs = plt.subplots(2,2,figsize=(12,9))
for i, source in enumerate(datasets):
    coefs = pd.read_parquet(f'parquet/{outcome}/Individual_Lasso_coefs_{source}_{outcome}.parquet')
    row, col = divmod(i,2)
    ax = axs[row,col]
    features = coefs['feature names'].str.split(r'numeric__|categorical__sex_', expand=True)
    color_palette = []
    for color_indice in coefs['color']: # assign colour corresponding to the color indice
        if color_indice == 1:
            color_palette.append('b')
        else: 
            color_palette.append('r')
    sns.barplot(x=coefs["abs_coefs"].iloc[:10], y=features[1].iloc[:10], hue=features[1].iloc[:10], orient="h", palette=color_palette[:10], legend=False, alpha=0.5, ax=ax)
    ax.set_xlabel("Absolute Value of Coefficient", fontsize=15)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=16, size=0)  
    ax.set_title(source, fontsize=16)
plt.tight_layout()
plt.savefig(f'images/barplots/{outcome}/Individual_Lasso_coefs_{outcome}.png')
plt.show()