import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from magging import Magging

"""
This script provides diagnostic plots and functions for analyzing Magging models:

1. SqrtAbsStandardizedResid: Plots the standardized residuals after applying a square root
 and absolute transformation.

2. CookDistance: Computes and plots Cook's Distance for each observation.

3. QQPlot: Generates QQ plots for the model's residuals to check for normality.

4. TukeyAnscombe: Creates Tukey-Anscombe plots for analyzing residuals.
"""

def SqrtAbsStandardizedResid(model, Xy):
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    plt.figure(figsize=(10, len(model.groups)))
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        transformed_residuals = np.sqrt(np.abs(residues / np.std(residues)))
        plt.scatter(np.arange(len(transformed_residuals)), transformed_residuals, marker='o')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title("Standardized Residuals Plot")
        plt.xlabel("Observation Index")
        plt.ylabel("Standardized Residuals")
        plt.grid(True)
        plt.show()

def CookDistance(model, Xy): 
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    plt.figure(figsize=(10, len(model.groups)))
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        n = len(residues)
        cook_model = sm.OLS(residues, np.ones(n)).fit()
        influence = cook_model.get_influence()
        cooks_distance = influence.cooks_distance[0]

        # Create a Cook's distance plot
        plt.stem(cooks_distance, markerfmt='ro', linefmt='-', basefmt=' ')
        plt.title("Cook's Distance Plot")
        plt.xlabel("Observation Index")
        plt.ylabel("Cook's Distance")
        plt.grid(True)
        plt.show()

def QQPlot(model, Xy):    
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    # Create a list to store the QQ plot for each parameter's residues
    plt.figure(figsize=(10, len(model.groups)))
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))

        # Create the QQ plot
        sm.qqplot(residues, line='s')
        plt.title(f'QQ Plot for {group}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.grid(True)
    plt.show()

def TukeyAnscombe(model, Xy):
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    # Create a list to store the QQ plot for each parameter's residues
    plt.figure(figsize=(10, len(model.groups)))
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        tukey_anscombe = (residues - np.mean(residues)) / np.std(residues)

        plt.scatter(np.arange(len(tukey_anscombe)), tukey_anscombe)
        plt.title(f'Tukey-Anscombe Plot for Group {group}')
        plt.xlabel('Data Point Index')
        plt.ylabel('Tukey-Anscombe Value')
        plt.legend()
        plt.grid(True)
        plt.show()