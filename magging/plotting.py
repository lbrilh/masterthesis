import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import os

category = 'age_group'
y = 'outcome'
age_group = True
dataset = 'mimic'

# Load data from global parquet folder 
def load_data(outcome, source, version='train'):
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    relative_path = os.path.join('..', 'Parquet', f'{outcome}_data_{source}_{version}.parquet')
    file_path = os.path.abspath(os.path.join(current_directory, relative_path))
    _data = pd.read_parquet(file_path, engine='pyarrow')
    return _data 

# Include only admissions with recorded sex
_Xydata={
    'eicu': load_data('hr','eicu')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'hirid': load_data('hr','hirid')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'mimic': load_data('hr','mimic')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))],
    'miiv': load_data('hr','miiv')[lambda x: (x['sex'].eq('Male'))|(x['sex'].eq('Female'))]
}

if age_group:
    for dataset in ['eicu', 'hirid', 'miiv', 'mimic']:
        bins = [0, 15, 39, 65, float('inf')]
        labels = ['child', 'young adults', 'middle age', 'senior']

        # Use pd.cut to create a new 'age_group' column
        _Xydata[dataset]['age_group'] = pd.cut(_Xydata[dataset]['age'], bins=bins, labels=labels, right=False)

else: 
    _Xydata['mimic'][category].fillna(value='N/A', inplace=True)

print(_Xydata['mimic']['age_group'].isna().sum())


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))

plt.subplot(1,2,1)
plt.xticks([0, 50, 100, 150])
f = sns.boxplot(data=_Xydata['mimic'], x=y, hue=category, y=category, hue_order=['child', 'young adults', 'middle age', 'senior'],
                palette='crest', legend=False, fill=True, linewidth=1, boxprops=dict(alpha=0.6))
f.set_xlabel('',)
f.set_ylabel('',)
sns.set_style('ticks')
plt.subplot(1,2,2)
#fig.add_subplot(1,2,2, frameon=False)

ax = sns.kdeplot(data=_Xydata['mimic'], x='outcome', hue=category, common_norm=False, hue_order=['child', 'young adults', 'middle age', 'senior'],
            palette='crest', fill=True, linewidth=0, alpha=0.5, legend=False)

ax.set_xlabel('')
ax.set_ylabel('')

sns.set_style('ticks')
sns.despine()
#ax.set_title(f'Density of average hr 48-72h after admission categorized by age in {dataset}', fontdict={'fontweight': 'bold'})

#sns.move_legend(ax, 'upper left', title=None)
plt.suptitle(f'Average hr 48-72h after admission categorized by age in {dataset}', fontweight= 'bold', fontsize=15)
fig.supxlabel('average hr 48-72h')
plt.setp(ax.spines.values(),linewidth=1)
plt.setp(f.spines.values(),linewidth=1)
plt.show()