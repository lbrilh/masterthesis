'''
    This file contains the implementation of the preprocessing.
'''

import warnings

import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def make_feature_preprocessing(missing_indicator=True, categorical_indicator=True):
    """Make preprocessing for features."""
    if categorical_indicator: 
        preprocessors = [
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            )
        ]
    else:
            preprocessors = [
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                FunctionTransformer(lambda x: x.astype("category")),
                CATEGORICAL_COLUMNS,
            )
        ]
    
    if missing_indicator:
        preprocessors += [
            ("missing_indicator", MissingIndicator(features="all"), NUMERICAL_COLUMNS)
        ]

    return preprocessors