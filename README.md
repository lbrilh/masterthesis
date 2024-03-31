# External Validity of Predictors in ICU Data

Welcome to the GitHub repository for my master's thesis.

## Structure

This repository is structured as follows:

- **Feature Selection**: This folder contains scripts for comparing feature selection methods to determine which ones enhance the distinguishability between shared effects across datasets and dataset-specific coefficients. This folder corresponds to Chapter 5.4 (Data Identification).
  
- **Fine Tuning**: This folder contains scripts to employ a fine-tuning approach using a subset of test data for hyperparameter selection. It includes implementations of custom models like LGBM Boosted Anchor and refitted Random Forest. This folder corresponds to Chapter 3 (Fine-tuning approach). 
  
- **Forward Selection**: This folder includes the implementation of Stepwise Regression for Data Shared Lasso and its robustness on test data, as a function of the degree of sharing parameter and regularization strength. This folder corresponds to Chapter 6 (Forward Selection). It also contains the application of DSL to ICU data (c.f. Chapter 5.3).

- **Magging**: Here, you'll find implementation of Magging. It includes a comparison of Magging and Data Shared Lasso on artificial test data. This folder corresponds to Chapter 4 (Magging) and Chapter 5.5 (DSL vs. Magging). 

- **LogTransform**: This folder contains the scripts to generate the histograms of the predictors and the log-transformed densities. This folder corresponds to Chapter 2.3 (Features and preprocessing).

Not part of the main work but utilized for functionality:

- **icu_experiments**: This folder is used for loading data stored in datasets, log-transformations, and other high-level preprocessing of the data (cf. Chapter 2.3 (Features and preprocessing)). Concepts such as log-transformed variables can be found in constants. Model-specific preprocessing is also utilized from here (c.f. preprocessing.py).

- **ivmodels**: This folder contains the implementation of Anchor Regression. 

icu_experiments and ivmodels are forked from Malte Londschien (https://github.com/mlondschien/).

A detailed overview of the files contained in each folder can be found in the folder-specific `README.md`.

### Usage

Before running the scripts in the folders, please download the datasets using `ricu` (https://eth-mds.github.io/ricu/) and store them as Parquet files in a folder (in the same directory as this README.md), i.e. `data/processed/source_name` (c.f. DATA_PATH in icu_experiments/load_data.py).

Please install the corresponding Python environment using the `environment.yml` file. 

The datasets used in this work (eICU, HiRID, MIMIC-III, and MIMIC-IV) can be downloaded from [PhysioNet](https://physionet.org/).

#### Run Python jobs
Navigate to the directory of the chosen file.
Run: python chosen_file.py