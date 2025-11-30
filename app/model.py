"""Model utilities for Bayesian linear regression."""
from typing import Tuple

import numpy as np
from sklearn.linear_model import BayesianRidge


class BayesianLinearRegression:
    """Wrapper around scikit-learn's BayesianRidge with uncertainty helpers."""

    def __init__(self) -> None:
        self.model = BayesianRidge()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Bayesian ridge regression model."""
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict targets with associated uncertainty.

        Returns:
            A tuple containing predictive means and standard deviations.
        """
        mean, std = self.model.predict(X, return_std=True)
        return mean, std


__all__ = ["BayesianLinearRegression"]
