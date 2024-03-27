# Examining the Distributional Robustness of Statistical Learning Methods

Welcome to the GitHub repository for my master's thesis.

## Structure

This repository is structured as follows:

- **Feature Selection**: This section contains scripts for comparing feature selection methods and examining which ones lead to increased identifiability between shared effects among datasets and dataset-specific coefficients.
  
- **Fine Tuning**: This section involves employing a fine-tuning approach using a subset of test data for hyperparameter selection. It includes implementations of custom models like LGBM Boosted Anchor and refitted Random Forest. Additionally, it examines the behavior of Anchor Regression as a function of the regularization strength gamma.
  
- **Forward Selection**: This part includes the implementation of Stepwise Regression for Data Shared Lasso and its robustness on test data, as a function of the degree of sharing parameter and regularization strength.

- **Magging**: Here, you'll find implementations of linear and non-linear Magging estimators. Additionally, it includes a comparison of Magging and Data Shared Lasso on artificial test data.

- **Random**: This section houses histograms of predictors and storage of results that are not part of the main work.

Not part of the main work but utilized for functionality:

- **icu_experiments**: This folder is used for loading data stored in datasets, log-transformations, and other high-level preprocessing of the data (cf. Chapter 2.1 preprocessing). Concepts such as log-transformed variables can be found in constants. Model-specific preprocessing is also utilized from here (in preprocessing.py).

- **ivmodels**: This folder contains the implementation of Anchor Regression. 

icu_experiments and ivmodels are forked from Malte Londschien (https://github.com/mlondschien/).

A detailed overview of the files contained in each folder can be found in the folder-specific `README.md`.

### Usage

Before utilizing the scripts, please download the datasets using `ricu` (https://eth-mds.github.io/ricu/) and store them as Parquet files in a folder (in the same directory as this README.md) like this: `data/processed/source_name` (c.f. DATA_PATH in icu_experiments/load_data.py).

Please install the corresponding environment using the `environment.yml` file.

The datasets used in this work (eICU, HiRID, MIMIC-III, and MIMIC-IV) can be downloaded from [PhysioNet](https://physionet.org/).