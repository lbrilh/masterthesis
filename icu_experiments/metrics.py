import numpy as np

METRICS = {
    "mse": lambda y, yhat: np.mean(np.square(yhat - y)),
    "mae": lambda y, yhat: np.mean(np.abs(yhat - y)),
    "quantile_08": lambda y, yhat: np.quantile(np.square(yhat - y), 0.8),
    "quantile_09": lambda y, yhat: np.quantile(np.square(yhat - y), 0.9),
    "quantile_095": lambda y, yhat: np.quantile(np.square(yhat - y), 0.95),
}


def compute_metrics(yhat, y, metrics=None):
    """Compute various regression metrics."""
    if metrics is None:
        metrics = METRICS.keys()

    return {metric: METRICS[metric](yhat, y) for metric in metrics}
