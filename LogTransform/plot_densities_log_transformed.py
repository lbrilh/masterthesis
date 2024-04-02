import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from icu_experiments.constants import (
    CATEGORICAL_COLUMNS,
    LOG_COLUMNS,
    NUMERICAL_COLUMNS,
    SOURCES,
)
from icu_experiments.load_data import load_data_for_prediction

OUTPUT_PATH = "plots/density_plots"

NCOLS = 5

SOURCE_COLORS = {
    "eicu": "black",
    "mimic": "red",
    "hirid": "blue",
    "miiv": "orange",
}

def main(x_offset):  # noqa D
#    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    Xy = pd.concat(
        [
            x[key]
            for x in load_data_for_prediction(X_offset=x_offset, outcome=None).values()
            for key in ["train", "test"]
        ]
    )

    # nrows = len(NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS) // NCOLS + 1
    nrows = len(LOG_COLUMNS) // NCOLS

    fig, axes = plt.subplots(nrows=nrows-1, ncols=NCOLS+1, figsize=(15, 5 * nrows))

    # for col, ax in zip(NUMERICAL_COLUMNS, axes.flat[: len(NUMERICAL_COLUMNS)]):
    for col, ax in zip(LOG_COLUMNS, axes.flat[: len(LOG_COLUMNS)]):
      
        x_min, x_max = Xy[col].dropna().quantile(0.001), Xy[col].dropna().quantile(
            0.999
        )

        for source in SOURCES:
            density = gaussian_kde(
                Xy[lambda x: x["source"] == source][col].dropna(),
                bw_method=1 / np.power(Xy[col].notna().sum(), 0.25),
            )
            linspace = np.linspace(x_min, x_max, num=100)
            ax.plot(
                linspace,
                density(linspace),
                #label=f"{source} ({100 * Xy[lambda x: x['source'] == source][col].isna().mean():.0f}%)",
                color=SOURCE_COLORS[source],
            )

            if col in LOG_COLUMNS:
                ax.set_title(f"log({col})", fontsize=20)
            else:
                ax.set_title(col)

            #ax.legend()
    plt.tight_layout()
    fig.savefig(f"plots/density_plots/{x_offset}.png")
    plt.close(fig)


if __name__ == "__main__":
    main(0)