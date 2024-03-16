## Directory Structure

- `ForwardSelection/`: Contains the main scripts for the feature selection process.
  - `baseline.py`: Implements the baseline model using forward feature selection with Lasso regression. It saves the results and selected hyperparameters.
  - `dsl.py`: Extended version of the feature selection using Data Shared Lasso (DSL) with additional preprocessing steps and multiple alpha values.
  - `plotting.py`: Generates visualizations for the results of the models, including the baseline and DSL versions.

- `Images/`: A directory meant for storing images.

- `baseline_results/`: Intended to store the results from the baseline model as `.parquet` files.

- `dsl_results/`: Intended to store the results from the DSL model as `.parquet` files.

- `preprocessing.py`: Defines the preprocessing steps for handling categorical and numerical data, including imputation, scaling, and encoding.

- `constants.py`: Defines categorical and numerical columns.
