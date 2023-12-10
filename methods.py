from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from ivmodels.anchor_regression import AnchorRegression


class AnchorBoost(BaseEstimator, RegressorMixin):
    def __init__(self, anchor_params=None, lgbm_params=None):
        # Initialize parameters
        self.anchor_params = anchor_params if anchor_params is not None else {}
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}

    def fit(self, X, y):
        # Initialize and fit the Anchor Regression model
        self.anchor_model = AnchorRegression(**self.anchor_params)
        self.anchor_model.fit(X, y)

        # Calculate residuals
        residuals = y - self.anchor_model.predict(X)

        # Initialize and fit the LGBMRegressor with residuals
        self.lgbm_model = LGBMRegressor(**self.lgbm_params)
        self.lgbm_model.fit(X, residuals)

        return self

    def predict(self, X):
        # Check if fit has been called
        if not hasattr(self, 'anchor_model') or not hasattr(self, 'lgbm_model'):
            raise AttributeError("Models have not been fitted. Call fit() first.")

        # Make predictions
        anchor_predictions = self.anchor_model.predict(X)
        lgbm_predictions = self.lgbm_model.predict(X)

        # Combine predictions
        return anchor_predictions + lgbm_predictions
        

class CVMixin:
    """Mixin for models, adding a `fit_predict_cv` method."""

    def __init__(self, cv=5, **kwargs):
        super().__init__(**kwargs)
        self.cv = cv

    def fit_predict_cv(self, X, y):
        """
        Fit the model and predict the outcome using cross-validation.

        At the end, this fits the model itself on the whole data.
        """
        yhat = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in KFold(n_splits=self.cv).split(X):
            if isinstance(X, pd.DataFrame):
                X_train, y_train, X_val = (
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                    X.iloc[val_idx],
                )
            else:
                X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]

            self.fit(X_train, y_train)
            yhat[val_idx] = self.predict(X_val)

        self.fit(X, y)
        return yhat


class RefitLGBMRegressor(BaseEstimator):
    """
    LGBM Regressor that gets refit on new data.

    Parameters
    ----------
    prior : LGBMRegressor
        Prior model.
    decay_rate : float
        Decay rate for refitting. If `decay_rate=1`, the new data is ignored.
    """

    def __init__(self, prior=None, decay_rate=0.5):
        self.prior = prior
        self.decay_rate = decay_rate

    def fit(self, X, y):  # noqa D
        if not isinstance(self.prior, lgb.LGBMRegressor):
            raise ValueError("Prior must be a LGBMRegressor")

        self.model = copy.deepcopy(self.prior)
        new_booster = self.model.booster_.refit(
            data=X, label=y, decay_rate=self.decay_rate, n_jobs=1
        )
        self.model._Booster = new_booster
        return self

    def predict(self, X):  # noqa D
        return self.model.predict(X)


class RefitLGBMRegressorCV(CVMixin, RefitLGBMRegressor):
    """
    LGBM Regressor that gets refit on new data.

    Parameters
    ----------
    prior : LGBMRegressor
        Prior model.
    decay_rate : float
        Decay rate for refitting. If `decay_rate=0`, the new data is ignored.
    cv : int
        Number of folds for cross-validation.
    """

    def __init__(self, prior=None, decay_rate=0.5, cv=5):
        super().__init__(prior=prior, decay_rate=decay_rate, cv=cv)    