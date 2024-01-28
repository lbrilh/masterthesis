import copy
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression

class Magging(BaseEstimator):
""" Magging Estimator.

    The optimization objective for Magging is:

        
    
    The data is assumed to be in the format Xy, i.e the response variable is part of the data and
    it is assumed that there exist a column in Xy indicating the group. 

    Parameters
    ----------
    model : float, default=1.0
        The model to perform estimations in the groups.

    grouping_column : str
        Column used to determine groups in Xy.
    
    **kwargs :
        Additional arguments passed on to model.
    
    Attributes
    ----------
    models : dict
        A dictionary containing models fitted for each group.

    model : estimator object
        The base model used for fitting and making predictions in the groups.

    grouping_column : str
        Column used to determine groups in Xy.

    Xy : pandas.DataFrame
        The input DataFrame containing predictor matrix, response vector, and group column.

    groups : numpy.ndarray
        An array containing unique group values derived from the grouping_column.

    X : pandas.DataFrame
        The input predictor matrix.

    group_prediction_matrix : numpy.matrix
        A matrix containing group-wise predictions for the response variable.

    w : numpy.array
        An array containing magging weights calculated on the predictor matrix.

    groups_magging_dist : dict
        A dictionary containing magging distances for coefficient vectors from each group.

    magging_dist : float
        The magging distance for the estimated maximin coefficient vector (if applicable).

    y_pred : numpy.array
        The magging estimator for the response variable.

    Methods
    -------
    group_fit(Xy, name_response_var, min_values=8)
        Fit models for each group based on the input data.

    group_predictions(X)
        Make predictions for the response variable using the fitted models for each group.

    magging_distance(u, X=pd.DataFrame(), v=0)
        Calculate the magging distance for a given vector.

    weights(X)
        Calculate magging weights for the magging estimator.

    predict(X)
        Calculate the magging estimator for the response variable.
    """
    
    def __init__(self, model, grouping_column, **kwargs):
        self.models = {}
        self.model = model(**kwargs)
        self.grouping_column = grouping_column
        

    
    def group_fit(self, Xy, name_response_var, min_values = 8):
        """
        A model is fitted for each group.

        Parameters
        ----------
        Xy : pandas.DataFrame
            A DataFrame containing the predictor matrix, the response vector and a column indicating the groups.
        name_response_var : str
            The name of the response variable in Xy. 
        min_values : int
            The minimal number of observations in a group to fit a model. 

        Returns
        -------
        dict
            A dict containing the models.
        """

        if not isinstance(Xy, pd.DataFrame):
            raise ValueError('X must be a pd.DataFrame')
        
        if self.grouping_column not in Xy.columns:
            raise ValueError('Column to group X is not a column in X')
        
        self.Xy = Xy

        self.groups = Xy[self.grouping_column].unique()
      
        Xygrouped = Xy.groupby(by=self.grouping_column)

        for group_name, group_data in Xygrouped:
            if len(group_data) >= min_values:
                model = copy.copy(self.model)
                _Xgroup = group_data.drop(columns=[self.grouping_column, name_response_var])
                _ygroup = group_data[[name_response_var]]
                model.fit(_Xgroup, _ygroup)
                self.models[group_name] = model
            else: 
                self.models[group_name] = None
        return self
    
    def group_predictions(self, X):
        """
        The response vector for the predictor matrix X is predicted for each model.

        Parameters
        ----------
        X : pandas.DataFrame
            A DataFrame containing the predictor matrix.

        Returns
        -------
        dict
            A dict containing information about the predictions for each group.
        """

        self.X = X
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X must be a pd.DataFrame')
        if not self.models: 
            raise ValueError('No model fitted')
        group_predictions = []
        for group in self.groups:
            if group not in self.models:
                raise ValueError(f'No model fitted for group: {group}')
            else: 
                if self.models[group]:
                    print(group)
                    model = self.models[group]
                    model_predictions = model.predict(X)
                    group_predictions.append(model_predictions)
                else: 
                    print(f'Group {group} has not been fitted and thus is skipped for predictions')
        self.group_prediction_matrix = np.matrix(group_predictions).T
        if self.group_prediction_matrix.shape == (0,0):
            print('Warning: No groups have been used for predictions')
        return self.group_prediction_matrix


    def magging_distance(self, u, X=pd.DataFrame(), v=0):
        """
        Calculate the magging distance. 

        Parameters
        ----------
        u : numpy.array
            A vector to determine its magging distance. 

        X : pandas.DataFrame, default empty
            A DataFrame containing the predictor matrix. Default: Calculate the distance based on the 
            matrix used for predictions in the groups. 
        
        v : numpy.array, default = 0
            An additional vector. Needed if one wants to calculate the distance between to vectors.
            If no vector is provided, v is the origin. 

        Returns
        -------
        float
            The magging distance.
        """

        if not X:
            X = self.X
        Sigma = (X.T@X)/X.shape[0]
        u_array = np.array(u)
        v_array = np.array(v)
        return (u_array -  v_array).T @ Sigma @ (u_array - v_array)
    
    def weights(self, X):
        """
        Calculate the weights of the magging estimator. 

        The weights are rounded to four digits. Additionaly, calculates the magging distance for each coefficient vector from each group
        and the magging distance for the estimated maximin coefficient vector (if a LinearRegressor or Lasso is used for the predictions
        in each group).

        Parameters
        ----------
        X : pandas.DataFrame
            A DataFrame containing the predictor matrix.

        Returns
        -------
        float
            The weights of the magging estimator calculated on the predictor matrix.
        """

            # Set-up of the quadratic program to determine magging weights
            r = self.group_prediction_matrix.shape[1]
            if r == 1:
                print('Warning: Only one group exists!')
            fhat = self.group_prediction_matrix
            H = fhat.T @ fhat / X.shape[0]
            print(H.shape)

            if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
                print("Warning: Matrix H is not positive definite")
                H += 1e-5
            P = matrix(H)
            q = matrix(np.zeros(r))
            G = matrix(-np.eye(r))
            h = matrix(np.zeros(r))
            A = matrix(1.0, (1, r))
            b = matrix(1.0)

            # Solve the quadratic program to obtain magging weights
            solution = solvers.qp(P, q, G, h, A, b)
            self.w = np.array(solution['x']).round(4).flatten() # Magging weights

            if isinstance(self.model, (Lasso, LinearRegression)):
                coef = []
                self.groups_magging_dist = {}
                for group in self.groups:
                    model_coef = self.models[group].coef_
                    coef.append(model_coef)
                    self.groups_magging_dist[group] = self.magging_distance(model_coef)

                magging_coef = np.dot(np.matrix(coef).T, self.w).T
                self.magging_dist = self.magging_distance(magging_coef)

            print('Magging weights: ', self.w)

            return self.w

    def predict(self, X):
        """
        Calculate the magging estimator for the response variable. 

        Parameters
        ----------
        X : pandas.DataFrame
            A DataFrame containing the predictor matrix.
        
        Returns
        -------
        numpy.array
            The magging estimator. 
        """
        
        predictions = []
        number_of_groups = self.group_prediction_matrix.shape[1]

        for group in self.groups:
            model = self.models[group]
            if model: 
                predictions.append(model.predict(X))

        self.y_pred = np.dot(np.array(predictions).T, self.w)

        return self.y_pred


        





'''    
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
        return (u_array -  v_array).T @ Sigma @ (u_array - v_array)'''