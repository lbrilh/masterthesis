'''This file creates a figure containing a boxplot and a kernel-density estimator plot for 
the average hr 48-72h after admission categorized by age group. 
'''


import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import os

category = 'age_group'
y = 'outcome'

if category == 'age_group':
    age_group = True
else: 
    age_group = False

# Load data from parent parquet folder 
def load_data(outcome, source, version='train'):
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    relative_path = os.path.join('..', 'Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.abspath(os.path.join(current_directory, relative_path))
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 

# Load data and include only admissions with recorded sex
_Xydata={
    'eicu': load_data('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid': load_data('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic': load_data('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv': load_data('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}

if age_group:
    # Add category age_group to datasets
    bins = [0, 15, 39, 65, float('inf')]
    labels = ['child', 'young adults', 'middle age', 'senior']
    for dataset in ['eicu', 'hirid', 'miiv', 'mimic']:
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)
else:
    # Fill missing values in category column
    for dataset in ['eicu', 'hirid', 'miiv', 'mimic']:
        _Xydata[dataset][category].fillna(value='N/A', inplace=True)

# Population pie charts
for dataset in ['eicu', 'hirid', 'mimic', 'miiv']:
    _Xydata[dataset][category] = _Xydata[dataset][category].cat.remove_unused_categories()
    _dfcategories = _Xydata[dataset][category].cat.categories
    len_dataset = _Xydata[dataset][category].shape[0]
    labels = _Xydata[dataset][category].value_counts().sort_index()
    fig, ax = plt.subplots()
    #print(sns.color_palette('crest'))
    if dataset == 'eicu': 
        colors = [(0.48942421, 0.72854938, 0.56751036), (0.24929311, 0.56486397, 0.5586654), (0.11131735, 0.39155635, 0.53422678), (0.14573579, 0.29354139, 0.49847009)]
    elif dataset in ['hirid', 'miiv']: 
        colors = [(0.24929311, 0.56486397, 0.5586654),(0.15615526, 0.47913212, 0.54850526),(0.14573579, 0.29354139, 0.49847009)]
    elif dataset == 'mimic':
        colors = [(0.34892097, 0.64828676, 0.56513633), (0.24929311, 0.56486397, 0.5586654), (0.11131735, 0.39155635, 0.53422678), (0.14573579, 0.29354139, 0.49847009)]
    ax.pie(x=_Xydata[dataset][category].value_counts().sort_index(), labels=_dfcategories, colors=colors, 
           wedgeprops={"linewidth": 1, "edgecolor": "white", "alpha": 0.6},
           autopct=lambda p: int(p/100*len_dataset))
    plt.title(f'Population in {dataset}', fontdict={'fontweight': 'bold', 'fontsize': 15} )

for dataset in ['eicu', 'hirid', 'miiv', 'mimic']:
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    df_categories = _Xydata[dataset][category].cat.categories # ordered categories from DataFrame
    _Xydata[dataset][category]=_Xydata[dataset][category].cat.remove_unused_categories()

    # Boxplot
    plt.subplot(1,2,1)
    plt.xticks([0, 50, 100, 150])
    f = sns.boxplot(data=_Xydata[dataset], x=y, hue=category, y=category, hue_order=df_categories,
                    palette='crest', legend=False, fill=True, linewidth=1, boxprops=dict(alpha=0.6))
    f.set_xlabel('')
    f.set_ylabel('')
    sns.set_style('ticks')
    plt.subplot(1,2,2)

    # KDE Plot
    ax = sns.kdeplot(data=_Xydata[dataset], x='outcome', hue=category, common_norm=False, hue_order=df_categories,
                palette='crest', fill=True, linewidth=0, alpha=0.5, legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    sns.set_style('ticks')
    sns.despine()

    # Figure
    plt.suptitle(f'Average hr 48-72h after admission categorized by age in {dataset}', fontweight= 'bold', fontsize=15)
    fig.supxlabel('average hr 48-72h')
    plt.setp(ax.spines.values(),linewidth=1)
    plt.setp(f.spines.values(),linewidth=1)

plt.show()