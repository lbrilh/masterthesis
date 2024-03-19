import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outcome = 'hr'
datasets = ['eicu', 'mimic', 'miiv', 'hirid']
coefs = pd.read_parquet('dsl_coefs.parquet')

fig, axs = plt.subplots(2,2,figsize=(12,9))
for i, source in enumerate(datasets):
    row, col = divmod(i,2)
    ax = axs[row,col]
    selected_rows = coefs['feature names'].str.contains(source)
    source_coefs = coefs[selected_rows]
    source_features = source_coefs['feature names'].str.split(r'numeric_|categorical_sex_|__categorical', expand=True)
    color_palette = []
    for color_indice in source_coefs['color']: # assign colour corresponding to the color indice
        if color_indice == 1:
            color_palette.append('b')
        else: 
            color_palette.append('r')
    sns.barplot(x=source_coefs["abs_coefs"].iloc[:10], y=source_features[1].iloc[:10], hue=source_features[1].iloc[:10], orient="h", palette=color_palette[:10], legend=False, alpha=0.5, ax=ax)
    ax.set_xlabel("Absolute Value of Coefficient")
    ax.set_ylabel('')
    ax.set_title(source)
plt.tight_layout()
plt.savefig(f'images/barplots/DSL_individual_coefs_{outcome}.png')

shared = coefs['feature names'].str.contains('passthrough')
shared_coefs = coefs[shared]
source_features = shared_coefs['feature names'].str.split(r'passthrough_features__numeric__|passthrough_features__categorical__sex_',expand=True)
color_palette = []
for color_indice in shared_coefs['color']: # assign colour corresponding to the color indice
    if color_indice == 1:
        color_palette.append('b')
    else: 
        color_palette.append('r')


plt.figure(figsize=(12,9))
sns.barplot(x=shared_coefs["abs_coefs"].iloc[:10], y=source_features[1].iloc[:10], hue=source_features[1].iloc[:10], orient="h", palette=color_palette[:10], legend=False, alpha=0.5)
plt.ylabel('Shared Coefficients')
plt.xlabel("Absolute Value of Coefficient")
plt.savefig(f'images/barplots/DSL_shared_coefs_{outcome}.png')
plt.show()