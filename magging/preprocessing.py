'''
    This file contains the implementation of the preprocessing.
'''
import warnings
import pandas as pd

from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, FunctionTransformer, QuantileTransformer

from constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

def make_feature_preprocessing(outcome, grouping_column=None, missing_indicator=True, categorical_indicator=True, lgbm=False):
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
            ),
            (
                'outcome', 
                FunctionTransformer(func=lambda X: X),
                [outcome]
            ),
            (
                'grouping_column',
                FunctionTransformer(func=lambda X: X),#.fillna(value='N/A')), #use only when not age_group
                [grouping_column]
            )
        ]
    elif lgbm:
        preprocessors = [
            (
                "numeric",
                FunctionTransformer(func=lambda X: pd.DataFrame(X), validate=False),
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
            ),
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
            ),
        ]
    if missing_indicator:
        preprocessors += [
            ("missing_indicator", MissingIndicator(features="all"), NUMERICAL_COLUMNS)
        ]

    return preprocessors