import numpy as np
from glum import GeneralizedLinearRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold


class EmpiricalBayesRidge(GeneralizedLinearRegressor):
    r"""
    Empirical Bayes Ridge Regression with l2-prior around beta.

    This optimizes
    `1 / n_train ||y - X beta||^2 + alpha \sum_j ((beta_j - beta_prior_j) / beta_std)^2`
    over `beta`.

    Parameters
    ----------
    alpha : float
        Regularization parameter.
    prior : Scikit-learn regressor
        Prior.
    P2 : np.array
        Standard deviation of beta for each feature.
    fit_intercept : bool
        Whether to fit an intercept.
    """

    def __init__(self, alpha=None, prior=None, P2=None, fit_intercept=True):
        super().__init__(alpha=alpha, P2=P2, fit_intercept=fit_intercept)
        self.prior = prior

    def fit(self, X, y):  # noqa D
        if np.isfinite(self.alpha):
            y_tilde = y - self.prior.predict(X)
            super().fit(X, y_tilde)
        return self

    def predict(self, X):  # noqa D
        if np.isfinite(self.alpha):
            return super().predict(X) + self.prior.predict(X)
        else:
            return self.prior.predict(X)


class EmpiricalBayesRidgeCV(EmpiricalBayesRidge):
    """
    Empirical Bayes Ridge Regression with l2-prior around beta.

    Additionally to `EmpiricalBayesRidge`, this class has a `fit_predict_cv` method
    that performs cross-validation to predict the outcome.

    Parameters
    ----------
    alpha : float
        Regularization parameter.
    prior : Scikit-learn regressor
        Prior.
    P2 : np.array
        Standard deviation of beta for each feature.
    fit_intercept : bool
        Whether to fit an intercept.
    cv : int
        Number of folds for cross-validation.

    """

    def __init__(self, alpha=None, prior=None, P2="identity", fit_intercept=True, cv=5):
        super().__init__(alpha=alpha, prior=prior, P2=P2, fit_intercept=fit_intercept)
        self.cv = cv

    def fit_predict_cv(self, X, y):
        """
        Fit the model and predict the outcome using cross-validation.

        At the end, this fits the model itself on the whole data.
        """
        yhat = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in KFold(n_splits=self.cv).split(X):
            super().fit(X[train_idx], y[train_idx])
            yhat[val_idx] = super().predict(X[val_idx])

        super().fit(X, y)
        return yhat


class ElasticNetCV(GeneralizedLinearRegressor):
    """Elastic Net model with `fit_predict_cv` method."""

    def __init__(self, prior=None, alpha=None, l1_ratio=0, fit_intercept=True, cv=5):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept)
        self.prior = prior
        self.cv = cv

    def fit_predict_cv(self, X, y):
        """
        Fit the model and predict the outcome using cross-validation.

        At the end, this fits the model itself on the whole data.
        """
        yhat = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in KFold(n_splits=self.cv).split(X):
            super().fit(X[train_idx], y[train_idx])
            yhat[val_idx] = super().predict(X[val_idx])

        super().fit(X, y)
        return yhat


class DummyRegressor(BaseEstimator):
    """Dummy model."""

    def __init__(self):
        pass

    def fit(self, X, y):  # noqa D
        return self

    def predict(self, X):  # noqa D
        raise ValueError("DummyRegressor cannot predict")


class PriorPassthrough(BaseEstimator):
    """
    Model that passes through all calls to `predict` and `fit_predict_cv` to the prior.

    This is useful if you want to use the data from the test environment for tuning
    only.
    """

    def __init__(self, prior):
        self.prior = prior

    def fit(self, X, y):  # noqa D
        return self

    def predict(self, X):  # noqa D
        return self.prior.predict(X)

    def fit_predict_cv(self, X, y):  # noqa D
        return self.prior.predict(X)
