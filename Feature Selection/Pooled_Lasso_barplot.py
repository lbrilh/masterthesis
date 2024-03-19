import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outcome = 'hr'
datasets = ['eicu', 'mimic', 'miiv', 'hirid']
coefs = pd.read_parquet('Pooled_Lasso_coefs.parquet')

features = coefs['feature names'].str.split(r'numeric__|categorical__sex_',expand=True)
color_palette = []
for color_indice in coefs['color']: # assign colour corresponding to the color indice
    if color_indice == 1:
        color_palette.append('b')
    else: 
        color_palette.append('r')

plt.figure(figsize=(12,9))
sns.barplot(x=coefs["abs_coefs"].iloc[:10], y=features[1].iloc[:10], hue=features[1].iloc[:10], orient="h", palette=color_palette[:10], legend=False, alpha=0.5)
plt.title('Pooled Lasso')
plt.ylabel('')
plt.xlabel("Absolute Value of Coefficient")
plt.savefig(f'images/barplots/Pooled_Lasso_coefs_{outcome}.png')
plt.show()