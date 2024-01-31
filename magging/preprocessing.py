import warnings

import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, FunctionTransformer

from constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

FILL_VALUES = {
    "hospital_id": -1,
    "numbedscategory": "missing",
    "teachingstatus": "missing",
    "region": "missing",
}


# Complicated fillna(), as fillna() does not support filling with []
def _list_fillna(x, col, fill_value):
    mask = x[col].isna()
    x.loc[mask, col] = pd.Series([[fill_value]] * mask.sum(), index=x[mask].index)
    return x


class MyMultiLabelBinarizer(MultiLabelBinarizer):
    """MultiLabelBinarizer that can handle unknown classes and pandas DataFrames."""

    def fit_transform(self, X, y=None):  # noqa D
        super().fit(X.squeeze(axis=1))
        return pd.DataFrame(
            super().transform(X.squeeze(axis=1)), columns=self.classes_, index=X.index
        )

    def transform(self, X, y=None):  # noqa D
        with warnings.catch_warnings():  # Ignore unknown class(es) warning
            warnings.simplefilter("ignore")
            return pd.DataFrame(
                super().transform(X.squeeze(axis=1)),
                columns=self.classes_,
                index=X.index,
            )

    def inverse_transform(self, X, y=None):  # noqa D
        return super().inverse_transform(X.squeeze(axis=1))

    def fit(self, X, y=None):  # noqa D
        return super().fit(X.squeeze(axis=1))

    def set_output(self, *, transform=None):  # noqa D
        self._sklearn_output_config = {"transform": transform}
        return self


def make_categorical_preprocessing(columns=None):
    """Make categorical preprocessing, that is, one-hot encoding."""
    return [
        (
            f"anchor_{column}",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            [column],
        )
        for column in columns
    ]


def make_list_preprocessing(columns=None):
    """Make list preprocessing, that is, multi-label binarization."""
    return [
        (f"anchor_{column}", MyMultiLabelBinarizer(), [column]) for column in columns
    ]


def make_anchor_preprocessing(anchor_columns=None):
    """Make preprocessing for anchor columns."""
    column_transformer = []

    for anchor_column in anchor_columns:
        if anchor_column in ["caregiver", "adm_caregiver", "provider", "adm_provider"]:
            column_transformer += make_list_preprocessing([anchor_column])
        elif anchor_column in [
            "hospital_id",
            "services",
            "insurance",
            "ethnic",
            "year",
            "hosp_adm_dow",
            "icu_adm_dow",
            "source",
        ]:
            column_transformer += make_categorical_preprocessing([anchor_column])
        else:
            raise ValueError(f"Unknown anchor column: {anchor_column}")

    return column_transformer


def make_feature_preprocessing(grouping_column, outcome, missing_indicator=True, categorical_indicator=True, lgbm=False):
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
            )
        ]
    if missing_indicator:
        preprocessors += [
            ("missing_indicator", MissingIndicator(features="all"), NUMERICAL_COLUMNS)
        ]

    return preprocessors