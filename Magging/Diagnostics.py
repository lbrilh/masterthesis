"""
    This script computes diagnostic plots and functions for analyzing Magging models:
        1. SqrtAbsStandardizedResid: Plots the standardized residuals after applying a square root
        and absolute transformation.   
        2. CookDistance: Computes and plots Cook's Distance for each observation.
        3. QQPlot: Generates QQ plots for the model's residuals to check for normality.
        4. TukeyAnscombe: Creates Tukey-Anscombe plots for analyzing residuals.
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from magging import Magging

def SqrtAbsStandardizedResid(model, Xy):
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    if len(model.groups)%2 == 1:
        nrows = len(model.groups)//2+1
    else: 
        nrows = len(model.groups)//2
    ncols = 2
    fig, axes = plt.subplots(nrows,ncols)
    i = 1
    for group in model.groups:
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        transformed_residuals = np.sqrt(np.abs(residues / np.std(residues)))
        ax = plt.subplot(nrows, ncols, i)
        plt.scatter(np.arange(len(transformed_residuals)), transformed_residuals, marker='o', alpha=0.05)
        ax.spines[['right','top']].set_visible(False)
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        plt.title(group)
        i += 1 
    if len(model.groups)%2 == 1:
        axes.flat[-1].set_visible(False)
    fig.suptitle("Standardized Residuals Plot", fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.show()

def CookDistance(model, Xy): 
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    if len(model.groups)%2 == 1:
        nrows = len(model.groups)//2+1
    else: 
        nrows = len(model.groups)//2
    ncols = 2
    fig, axes = plt.subplots(nrows,ncols)
    i = 1
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        n = len(residues)
        cook_model = sm.OLS(residues, np.ones(n)).fit()
        influence = cook_model.get_influence()
        cooks_distance = influence.cooks_distance[0]
        ax = plt.subplot(nrows, ncols, i)
        markerline, stemlines, baseline = plt.stem(cooks_distance, basefmt=' ')
        plt.setp(stemlines, visible=False)
        markerline.set_markeredgecolor((0,0,0,0))
        markerline.set_markerfacecolor((50/255,109/255,168/255,0.05))
        ax.spines[['right','top']].set_visible(False)
        plt.title(group)
        i += 1 
    if len(model.groups)%2 == 1:
        axes.flat[-1].set_visible(False)
    fig.suptitle("Cook's Distance Plot", fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.show()

def QQPlot(model, Xy):    
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    if len(model.groups)%2 == 1:
        nrows = len(model.groups)//2+1
    else: 
        nrows = len(model.groups)//2
    ncols = 2
    fig, axes = plt.subplots(nrows,ncols)
    i = 1
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        ax=plt.subplot(nrows, ncols, i)
        sm.qqplot(residues, line='s', ax=ax)
        plt.yticks([-100,0,100])
        plt.title(group)
        i+=1
    plt.suptitle(f'QQ Plot', fontweight='bold', fontsize=15)
    if len(model.groups)%2 == 1:
        axes.flat[-1].set_visible(False)
    plt.tight_layout()
    plt.show()

def TukeyAnscombe(model, Xy):
    if not isinstance(model, Magging):
        raise ValueError('The provided model is not a magging model')
    if len(model.groups)%2 == 1:
        nrows = len(model.groups)//2+1
    else: 
        nrows = len(model.groups)//2
    ncols = 2
    fig, axes = plt.subplots(nrows,ncols)
    i = 1
    for group in model.groups: 
        _Xygroup = Xy[Xy[model.grouping_column] == group]
        _Xgroup = _Xygroup.drop(columns=[model.grouping_column, model.name_response_var])
        _ygroup = np.array(_Xygroup[[model.name_response_var]]).reshape(-1)
        residues = np.array(_ygroup - model.models[group].predict(_Xgroup))
        tukey_anscombe = (residues - np.mean(residues)) / np.std(residues)
        ax = plt.subplot(nrows, ncols, i)
        plt.scatter(np.arange(len(tukey_anscombe)), tukey_anscombe, alpha=0.05)
        plt.yticks([-5,0,5])
        ax.spines[['right','top']].set_visible(False)
        plt.title(group)
        i += 1 
    if len(model.groups)%2 == 1:
        axes.flat[-1].set_visible(False)
    fig.suptitle("Tukey-Anscombe Plot for Group", fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.show()

def CorrelationPlot(yhat, y):
    plt.scatter(yhat, y, alpha=0.2)
    plt.title('Correlation of estimated and true y', fontweight='bold', fontsize=15)
    plt.xlabel('Predictions')
    plt.ylabel('y')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()