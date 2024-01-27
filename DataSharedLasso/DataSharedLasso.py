import numpy as np
from scipy.linalg import block_diag
import pandas as pd
from preprocessing import make_feature_preprocessing
from constants import NUMERICAL_COLUMNS
from sklearn.linear_model import Lasso

# Assume X is your original data matrix with observations as rows and features as columns
# y is the response vector
# groups is an array-like structure indicating the group of each observation
# r is a dictionary or array with scaling factors for the groups

def create_augmented_data(_Xydata, pipeline, group_column_name, r=0):
    unique_groups = _Xydata[group_column_name].unique()

    if r ==0:
        r = 1/sqrt(len(unique_groups))
    
    # Initialize lists to store the scaled blocks and the unscaled group data
    scaled_blocks = []
    unscaled_group_data = []
    y_tilde = []
    
    # Create the scaled block diagonal matrix and the unscaled group matrix
    for j in unique_groups:
        # Select the data for the current group
        group_data = pipeline.named_steps['preprocessing'].fit_transform(_Xydata[_Xydata[group_column_name] == j])
        group_y = _Xydata[_Xydata[group_column_name] == j]['outcome']
        
        # Scale the group-specific data by the corresponding r value
        if r.shape[0] == 1:
            scaled_group_data = group_data * r
        else: 
            scaled_group_data = group_data * r[j]
        
        # Append the scaled group data to the list for block diagonal
        scaled_blocks.append(scaled_group_data)
        
        # Append the unscaled group data to the list for vertical stacking
        unscaled_group_data.append(group_data)
        
        # Extend the y_tilde vector with the group's responses
        y_tilde.extend(group_y)
    
    # Use block_diag to create a block diagonal matrix from the scaled blocks
    scaled_block_matrix = block_diag(*scaled_blocks)
    
    # Stack the unscaled group data vertically
    unscaled_group_matrix = np.vstack(unscaled_group_data)
    
    # Concatenate the unscaled group matrix with the scaled block matrix horizontally
    Z = np.hstack([unscaled_group_matrix, scaled_block_matrix])
    
    return Z, np.array(y_tilde)