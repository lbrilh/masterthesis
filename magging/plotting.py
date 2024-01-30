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

plt.subplot(1,2,1)
sns.set_theme(font_scale=1.25)
sns.set_style('ticks')
sns.despine()

f = sns.catplot(data=_Xydata['mimic'], x=category, hue=category, y=y, kind='box', hue_order=['child', 'young adults', 'middle age', 'senior'],
                palette='crest', legend=False)
f.set_xlabels('')
f.set_titles(f'Average hr 48-72h after admission categorized by age in {dataset}', fontdict={'fontweight': 'bold'})

plt.subplot(1,2,2)

ax = sns.kdeplot(data=_Xydata['mimic'], x='outcome', hue=category, common_norm=False, hue_order=['child', 'young adults', 'middle age', 'senior'],
            palette='crest', fill=True)

ax.set_xlabel('average hr 48-72h')
sns.set_style('ticks')
sns.despine()
#ax.set_title(f'Density of average hr 48-72h after admission categorized by age in {dataset}', fontdict={'fontweight': 'bold'})

sns.move_legend(ax, 'upper left', title=None)
plt.suptitle(f'Average hr 48-72h after admission categorized by age in {dataset}', fontweight= 'bold')

plt.show()