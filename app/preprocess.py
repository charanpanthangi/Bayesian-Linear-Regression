"""Preprocessing utilities for splitting and scaling data."""
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_test_split_and_scale(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split the data into train and test sets and scale features.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of test data.
        random_state: Seed for reproducibility.

    Returns:
        Scaled train/test feature matrices, train/test targets, and fitted scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


__all__ = ["train_test_split_and_scale"]
