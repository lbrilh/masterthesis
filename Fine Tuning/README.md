# Fine Tuning

This directory contains scripts and resources for fine-tuning statistical learning models, focusing on regression models and the application of hyperparameter optimization techniques.

## Contents

- `experiments_script.py`: Main script for running experiments with different hyperparameter settings, employing a fine-tuning approach using a subset of test data for hyperparameter selection.
- `methods.py`: Includes the implementation of custom regression models such as `AnchorBoost` and `RefitLGBMRegressor` with added cross-validation functionality.
- `plotting.py`: Visualizes results, specifically plotting the mean squared error against the number of fine-tuning data points and gamma values for Anchor Regression.
- `set_up.py`: Configures the parameters and settings for experiments, including model selection, outcome variables, and hyperparameter grids.

### Subdirectories

- `__pycache__`: Python cache files.
- `parquet`: Contains Parquet files for data storage.
- `plots`: Stores generated plot images.

We use preprocessing.py and constants.py from icu_experiments.