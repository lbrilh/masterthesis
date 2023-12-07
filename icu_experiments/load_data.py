from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from icu_experiments.constants import (
    LIST_COLUMNS,
    LOG_COLUMNS,
    NUMERICAL_COLUMNS,
    RENAMINGS,
    SOURCES,
)

DATA_PATH = Path(__file__).parents[1] / "data" / "processed"


# Complicated fillna(), as fillna() does not support filling with []
def _list_fillna(x, col, fill_value):
    mask = x[col].isna()
    x.loc[mask, col] = pd.Series([[fill_value]] * mask.sum(), index=x[mask].index)
    return x


def load_data(source, cache=True):
    """
    Load ICU data from a single source.

    The data are aligned to have the same schema.

    Parameters
    ----------
    source : str
        One of "eicu", "mimic", "hirid", "miiv", "eicu_demo", "mimic_demo", "aumc".

    Returns
    -------
    dynamic : pd.DataFrame
        Dynamic data, i.e., data that changes over time. Columns `stay_id` and
        `time` are unique identifiers.
    static : pd.DataFrame
        Static data, i.e., data that is entered at admission. The column `stay_id`
        is a unique identifier.
    """
    dynamic_cached_path = DATA_PATH / source / "dyn_cache.parquet"
    static_cached_path = DATA_PATH / source / "sta_cache.parquet"

    if cache and dynamic_cached_path.exists() and static_cached_path.exists():
        dynamic = pd.read_parquet(dynamic_cached_path)
        static = pd.read_parquet(static_cached_path)
        return dynamic, static

    dynamic = pd.read_parquet(DATA_PATH / source / "dyn.parquet")
    dynamic = dynamic.rename(columns=RENAMINGS.get(source, {})).assign(source=source)

    static = pd.read_parquet(DATA_PATH / source / "sta.parquet")
    static = static.rename(columns=RENAMINGS.get(source, {})).assign(source=source)

    if source in ["mimic", "miiv"]:
        static = static.drop_duplicates(subset="stay_id", keep="first")

    if source in ["mimic", "miiv"]:
        caregiver = pd.read_parquet(DATA_PATH / source / "caregiver.parquet")
        caregiver = caregiver.rename(columns=RENAMINGS[source]).assign(source=source)
        provider = caregiver.groupby(["stay_id", "time"])["provider"].apply(list)
        caregiver = caregiver.groupby(["stay_id", "time"])["caregiver"].apply(list)

        dynamic = pd.merge(
            left=dynamic,
            right=caregiver.to_frame(name="caregiver"),
            left_on=["stay_id", "time"],
            right_index=True,
            how="outer",
        )
        dynamic = pd.merge(
            left=dynamic,
            right=provider.to_frame(name="provider"),
            left_on=["stay_id", "time"],
            right_index=True,
            how="outer",
        )

        adm_caregiver = pd.read_parquet(DATA_PATH / source / "adm_caregiver.parquet")
        adm_caregiver = adm_caregiver.rename(columns=RENAMINGS[source])
        grouped = adm_caregiver.groupby(["stay_id"])
        adm_caregiver = grouped["adm_caregiver"].apply(list)
        adm_provider = grouped["adm_provider"].apply(list)

        static = pd.merge(
            left=static,
            right=adm_caregiver.to_frame(name="adm_caregiver"),
            left_on="stay_id",
            right_index=True,
            how="left",
        )
        static = pd.merge(
            left=static,
            right=adm_provider.to_frame(name="adm_provider"),
            left_on="stay_id",
            right_index=True,
            how="left",
        )

    else:
        dynamic = dynamic.assign(caregiver=np.nan, provider=np.nan)
        static = static.assign(adm_caregiver=np.nan, adm_provider=np.nan)

    if source in ["eicu", "eicu_demo"]:
        hospital = pd.read_parquet(DATA_PATH / source / "hospital.parquet").rename(
            columns={"hospitalid": "hospital_id"}
        )
        static = pd.merge(
            left=static,
            right=hospital,
            on="hospital_id",
            validate="m:1",
            how="left",
        )
    else:
        static = static.assign(numbedscategory=None, teachingstatus=None, region=None)

    if cache:
        dynamic.to_parquet(dynamic_cached_path)
        static.to_parquet(static_cached_path)

    return dynamic, static


def load_data_for_prediction(
    sources=None,
    X_offset=0,
    y_offset=2,
    outcome="hr",
    log_transform=True,
    hospital_threshold=10,
):
    """
    Load ICU data from multiple sources for a simple prediction task.

    Combines dynamic and static data from sources. (Logarithms of) Dynamic variables are
    taken at `X_offset` days and averaged (e.g., averaged over 0-24h after admission if
    `X_offset` = 0). Similarly, the outcome is taken at `y_offset` days and averaged.

    Parameters
    ----------
    sources : list of str, optional, default = None
        One or more of "eicu", "mimic", "hirid", "miiv", "eicu_demo", "mimic_demo",
        "aumc". If None, all non-demo sources are used.
    X_offset : int, optional, default = 0
        Offset in days for the input data.
    y_offset : int, optional, default = 2
        Offset in days for the output data.
    outcome : str, optional, default = "hr"
        Variable to be used as the outcome.
    log_transform : bool, optional, default = True
        Whether to log-transform particular variables. The list of variables to be
        log-transformed is defined in `icu_experiments.constants.LOG_COLUMNS`.

    Returns
    -------
    Xy : pd.DataFrame
        Data for prediction. The column `outcome` contains the outcome variable.
    """
    if sources is None:
        sources = SOURCES

    data = {}

    for source in sources:
        dynamic, static = load_data(source)
        # dynamic = pd.merge(dynamic, static[["stay_id", "los_icu"]], on="stay_id", how="inner")
        # dynamic = dynamic[lambda x: x["time"].dt.total_seconds() / 3600 / 24 <= x["los_icu"]]

        X = dynamic[lambda x: x["time"].dt.days == X_offset].copy()
        if log_transform:
            X[LOG_COLUMNS] = np.log(
                np.where(X[LOG_COLUMNS] > 0, X[LOG_COLUMNS], np.nan)
            )

        groupby = X.groupby(["source", "stay_id"])
        X_agg = groupby[list(set(NUMERICAL_COLUMNS) & set(X.columns))].agg(np.nanmean)

        if source in ["mimic", "miiv"]:
            for col in set(LIST_COLUMNS) & set(dynamic.columns):
                X_agg[col] = groupby[col].agg(lambda x: list(x.explode().unique()))
        else:
            X_agg[["caregiver", "provider"]] = np.nan

        X = X_agg.reset_index()
        if outcome is not None:
            y = dynamic[lambda x: x["time"].dt.days == y_offset]
            if log_transform and outcome in LOG_COLUMNS:
                y[outcome] = np.log(np.where(y[outcome] > 0, y[outcome], np.nan))

            y = y.groupby("stay_id")[outcome].agg(np.nanmean)[lambda x: x.notnull()]

            Xy = X.merge(
                y,
                on="stay_id",
                validate="1:1",
                how="inner",
                suffixes=("", "_y"),
            ).rename(columns={f"{outcome}_y": "outcome"}, copy=False)
        else:
            Xy = X

        # static = static[lambda x: x["los_icu"] >= y_offset + 1]
        Xy = Xy.merge(static, on=["source", "stay_id"], validate="1:1", how="inner")

        # Need to fill with numeric, as MultiLabelBinarizer() internally sorts the labels
        Xy = _list_fillna(Xy, "caregiver", -1)
        Xy = _list_fillna(Xy, "provider", "missing")
        Xy = _list_fillna(Xy, "adm_caregiver", -1)
        Xy = _list_fillna(Xy, "adm_provider", "missing")

        if source in ["eicu", "eicu_demo"]:
            large_hospitals = Xy.groupby("hospital_id").size() >= hospital_threshold
            Xy = Xy[
                lambda x: x["hospital_id"].isin(large_hospitals[large_hospitals].index)
            ]

        data[source] = {}
        Xy_train, Xy_test = train_test_split(Xy, test_size=1600, random_state=0)
        data[source]["train"] = Xy_train
        data[source]["test"] = Xy_test

    return data


def load_task(task=None):
    """Load data for a particular task."""
    if task == "kidney":
        return load_data_for_prediction(y_offset=1, outcome="crea")
    else:
        return load_data_for_prediction(outcome=task)
