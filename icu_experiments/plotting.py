from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "anchor_regression__empirical_bayes_diagonal": "orange",
    "anchor_regression__empirical_bayes_identity": "red",
    "elastic_net__passthrough": "gray",
    "anchor_regression__passthrough": "blue",
    "dummy__elastic_net": "green",
}

LEGEND = {
    "anchor_regression__empirical_bayes_diagonal": "anchor + emp. bayes (diag)",
    "anchor_regression__passthrough": "anchor regression",
    "anchor_regression__empirical_bayes_identity": "anchor + emp. bayes (id)",
    "elastic_net__passthrough": "elastic net on source",
    "dummy__elastic_net": "elastic net on target",
}


def get_df_for_plotting(
    results: pd.DataFrame,
    metric: str,
    by: Optional[str] = None,
):
    """
    Aggregate results by method, target, n_test, sample_seed.

    For each group, select the row with the lowest (optimal) value in `metric`.
    """
    group_columns = ["method", "target", "n_test", "sample_seed"]

    if by is not None:
        group_columns.append(by)

    results[f"{metric}_min"] = results.groupby(group_columns)[metric].transform("min")
    results = results[lambda x: x[metric].eq(x[f"{metric}_min"])]
    results = results.drop(columns=f"{metric}_min")

    return results.sort_values(group_columns)


def plot_tuning(
    df_cv: Optional[pd.DataFrame],
    df_oracle: Optional[pd.DataFrame],
    metric: str,
    methods: list,
):
    """Plot the results of a tuning experiment."""
    targets = ["mimic", "miiv", "hirid", "aumc"]
    fig, axes = plt.subplots(1, len(targets), figsize=(12, 6))

    for target, ax in zip(targets, axes):
        for method in methods:
            if df_cv is not None:
                xy = (
                    df_cv[lambda x: x["method"].eq(method) & x["target"].eq(target)]
                    .groupby("n_test")[f"test_{metric}"]
                    .mean()
                    .sort_index()
                    .reset_index()
                )
                ax.plot(
                    xy["n_test"],
                    xy[f"test_{metric}"],
                    label=LEGEND[method],
                    color=COLORS[method],
                    alpha=0.8,
                )
            if df_oracle is not None:
                xy = (
                    df_oracle[lambda x: x["method"].eq(method) & x["target"].eq(target)]
                    .groupby("n_test")[f"test_{metric}"]
                    .mean()
                    .sort_index()
                    .reset_index()
                )
                ax.plot(
                    xy["n_test"],
                    xy[f"test_{metric}"],
                    color=COLORS[method],
                    linestyle="--" if df_cv is not None else None,
                    alpha=0.8,
                    label=f"{LEGEND[method]} (oracle)" if df_cv is None else None,
                )

        ax.set_xlabel("n_test")
        ax.set_xscale("log")
        ax.set_title(target)

    axes[0].set_ylabel(metric)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    fig.suptitle(metric)
    return fig
