'''
    Generate bar plots for the coefficients of Pooled Lasso.
    The response variable is "outcome".
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outcome = 'map'
datasets = ['eicu', 'mimic', 'miiv', 'hirid']

coefs = pd.read_parquet(f'parquet/{outcome}/Pooled_Lasso_coefs_{outcome}.parquet')
features = coefs['feature names'].str.split(r'numeric__|categorical__sex_',expand=True)

color_palette = []
for color_indice in coefs['color']: # assign colour corresponding to the color indice
    if color_indice == 1:
        color_palette.append('b')
    else: 
        color_palette.append('r')

plt.figure(figsize=(12,9))
sns.barplot(x=coefs["abs_coefs"].iloc[:10], y=features[1].iloc[:10], hue=features[1].iloc[:10], orient="h", palette=color_palette[:10], legend=False, alpha=0.5)
plt.ylabel('')
plt.tick_params(axis='y', labelsize=16, size=0)
plt.xlabel("Absolute Value of Coefficient", fontsize=15)
plt.savefig(f'images/barplots/{outcome}/Pooled_Lasso_coefs_{outcome}.png')
plt.show()