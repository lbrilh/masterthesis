"""
    Generating the histograms for the predictors in our datasets.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

from icu_experiments.load_data import load_data_for_prediction

import matplotlib.pyplot as plt

datasets = ['eicu', 'mimic', 'miiv', 'hirid']

numeric = [
    "age",
    "alb",
    "alp",
    "alt",
    "ast",
    "be",
    "bicar",
    "bili",
    "bnd",
    "bun",
    "ca",
    "cai",
    "ck",
    "ckmb",
    "cl",
    "crea",
    "crp",
    "dbp",
    "fgn",
    "fio2",
    "glu",
    "height",
    "hgb",
    "hr",
    "k",
    "lact",
    "lymph",
    "map",
    "mch",
    "mchc",
    "mcv",
    "methb",
    "mg",
    "na",
    "neut",
    "o2sat",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "resp",
    "sbp",
    "temp",
    "tnt",
    "urine",
    "wbc",
    "weight",
]

df = load_data_for_prediction(log_transform=False)
for source in datasets:
    fig, axs = plt.subplots(7,7, figsize=(21,21))
    df[source]['train'][numeric].hist(ax=axs, bins=30, alpha=0.5)
    for ax in axs.flat:
        ax.title.set_fontsize(20)  # Adjust the font size as needed
    plt.tight_layout()
    plt.savefig(f'plots/hist_{source}.png')
plt.show()
