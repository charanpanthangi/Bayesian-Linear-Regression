"""Data loading utilities for the diabetes dataset."""
from typing import Tuple

import numpy as np
from sklearn import datasets


def load_diabetes_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the scikit-learn diabetes dataset.

    Returns:
        Tuple containing feature matrix ``X`` and target vector ``y``.
    """
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    return X, y


__all__ = ["load_diabetes_data"]
