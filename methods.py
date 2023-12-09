from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from ivmodels.anchor_regression import AnchorRegression

class CustomizedAnchor(BaseEstimator, RegressorMixin):
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
    