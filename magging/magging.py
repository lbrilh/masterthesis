import os 
import re
import pandas as pd
import numpy as np
import joblib
from cvxopt import matrix, solvers
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def group_predictions(_Xytrain, _Xy_source, grouping_column, hyper_parameters, pipeline, estimators_folder):
    """
    Group predictions based on training data and hyperparameter tuning.

    Parameters
    ----------
    _Xytrain : pandas.DataFrame
        A DataFrame containing the predictor matrix and response vector of the training data.
    _Xy_source : str
        The source of the response-predictor matrix _Xytrain.
    grouping_column : str
        Column used to determine groups in _Xytrain.
    hyper_parameters : dict
        The hyperparameter dictionary.
    pipeline : sklearn.pipeline
        The pipeline to perform estimations.
    estimators_folder : str
        The folder where the estimator files will be stored.

    Returns
    -------
    dict
        A dict containing information about the grouped predictions and models.
    """

    # The dictionary to store the prediction vectors and file paths of models for each group
    _models = {}

    _Xygrouped=_Xytrain.groupby(by=grouping_column)

    _Xytrain = _Xytrain.groupby(by=grouping_column).apply(lambda x: x.reset_index(drop=True))

    for group_name, group_data in _Xygrouped:
        
        # Skip all groups with less than 8 observations
        if len(group_data)<8:
            _models[group_name]=None

        else: 

            search = GridSearchCV(pipeline, param_grid= {'model__' + key : value for key, value in hyper_parameters.items()})

            _ydata_group = group_data['outcome'].values
            search.fit(group_data, _ydata_group)

            _ypredict = search.predict(_Xytrain)

            sanitized_group_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(group_name))
            estimator_filename = f'best_estimator_{_Xy_source}_{sanitized_group_name}.joblib'
            estimator_file_path = os.path.join(estimators_folder, estimator_filename)
            joblib.dump(search.best_estimator_, estimator_file_path)

            _models[group_name] = {
                f'{grouping_column}': group_data[grouping_column].iloc[0],
                'best_params': search.best_params_,
                'cv_results': search.cv_results_,
                'estimated_coeff': search.best_estimator_.named_steps['model'].coef_,
                'y_predict': _ypredict,
                'residues': _ydata_group - search.predict(group_data),
                'model': estimator_file_path  # Store the filename instead of the model object
            }
    
    return _models


def weights(_Xytrain, _Xytrain_source, pipeline, parameters_file):
    """
    Calculate the weights of the magging estimator. 

    The weights are rounded to three digits. 

    Parameters
    ----------
    _Xytrain : pandas.DataFrame
        A DataFrame containing the predictor matrix and response vector of the training data.
    pipeline : sklearn.pipeline.Pipeline
        The scikit-learn pipeline used for predictions in magging. The preprocessing step in the pipeline must be named 'preprocessing'.
    parameters_file : str
        The path to the file where the estimators within each group are stored.

    Returns
    -------
    float
        The weights of the magging estimator calculated on the predictor matrix of _Xytrain. 
        This is returned only if there are more than one group; otherwise, it returns None.
    """

    # Load the parameter file for the regressor
    _dfmodels = pd.read_parquet(parameters_file)

    # Extract group predictions and skip groups with no calculated parameters
    _predictions=_dfmodels[_Xytrain_source].dropna().values 

    if len(_predictions)>1: # Avoid predictions on the entire dataset
                
        # Transform training data using a preprocessor
        _Xdata = pipeline.named_steps['preprocessing'].fit_transform(_Xytrain) # Maybe gram without preprocessing (missingness indicaator)

        # Set-up of the quadratic program to determine magging weights
        r=len(_predictions)
        fhat = np.column_stack([(_predictions[i])['y_predict'] for i in range(len(_predictions))])
        H = fhat.T @ fhat / _Xdata.shape[0]

        if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
            print("Attention: Matrix H is not positive definite")
            H += 1e-5
        
        P = matrix(H)
        q = matrix(np.zeros(r))
        G = matrix(-np.eye(r))
        h = matrix(np.zeros(r))
        A = matrix(1.0, (1, r))
        b = matrix(1.0)

        # Solve the quadratic program to obtain magging weights
        solution = solvers.qp(P, q, G, h, A, b)
        w = np.array(solution['x']).round(decimals=5).flatten() # Magging weights
        print('Magging weights: ', w)

        return w
    
    return None

def predict(weights, _Xytest, pipeline, parameters_file, train_source):
    """
    Calculate the magging estimator. 

    Parameters
    ----------
    weights: numpy.array
        An array containing the weights for each group. 
    _Xytest : pandas.DataFrame
        A DataFrame containing the predictor matrix and response vector of the test data.
    pipeline : sklearn.pipeline.Pipeline
        The scikit-learn pipeline used for predictions in magging. The preprocessing step in the pipeline must be named 'preprocessing'.
    parameters_file : str
        The path to the file where the estimators within each group are stored.
    
    Returns
    -------
    numpy.array
        The magging estimator. 
    """

    # Load the parameter file for the regressor
    _dfmodels = pd.read_parquet(parameters_file, engine='pyarrow')[train_source]

    # Extract group predictions and skip groups with no calculated parameters
    _predictions=_dfmodels.dropna().values

    # Transform test data using the same preprocessor
    _Xtarget = pipeline.named_steps['preprocessing'].fit_transform(_Xytest)

    # Make predictions and calculate MSE
    loaded_model_preds = []

    for i in range(len(_predictions)):

        # Load the stored model for group i 
        loaded_model = joblib.load(_predictions[i]['model'])

        ####################################################################################### This does not work (predict)

        # Calculate predictions using the loaded model
        y_pred_loaded = loaded_model.predict(_Xtarget)

        # Add the predictions to the list, weighted by the corresponding weights
        loaded_model_preds.append(weights[i] * y_pred_loaded)

    # Calculate the final prediction by summing up the weighted predictions
    y_pred = np.sum(loaded_model_preds, axis=0)

    return y_pred


def magging_distance(X_matrix, u, v):
    Sigma = (X_matrix.T@X_matrix)/X_matrix.shape[0]
    u_array = np.array(u)
    v_array = np.array(v)
    return (u_array -  v_array).T @ Sigma @ (u_array - v_array)