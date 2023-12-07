import json
import os
import tempfile

import mlflow
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Allow storing dict of numpy arrays as json.

    https://github.com/thepushkarp/til/blob/main/python/store-dictionary-with-numpy-arrays-as-json.md
    """

    def default(self, obj):  # noqa D
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def log_dict(dict, name):
    """Log a dictionary as json to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        with open(path, "w") as f:
            json.dump(dict, f, cls=NumpyEncoder)
        mlflow.log_artifact(path, target_dir)


def log_fig(fig, name):
    """Log a matplotlib figure to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        fig.savefig(path)
        mlflow.log_artifact(path, target_dir)


def log_df(df, name):
    """Log a pandas dataframe to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"

        if name.endswith(".csv"):
            df.to_csv(path, index=False)
        elif name.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unknown file extension: {name}")

        mlflow.log_artifact(path, target_dir)
