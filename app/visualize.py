"""Visualization helpers for Bayesian linear regression."""
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path | str
) -> None:
    """Plot actual vs. predicted targets and save to disk."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_uncertainty(
    X_axis: Iterable[float],
    mean: np.ndarray,
    std: np.ndarray,
    y_true: np.ndarray | None,
    output_path: Path | str,
    xlabel: str = "Sample index",
    ylabel: str = "Target",
) -> None:
    """Plot predictive mean with +/- 2*std uncertainty band."""
    plt.figure(figsize=(8, 5))
    X_axis = np.array(list(X_axis))
    plt.plot(X_axis, mean, label="Predictive mean")
    plt.fill_between(
        X_axis, mean - 2 * std, mean + 2 * std, color="orange", alpha=0.3, label="95% approx CI"
    )
    if y_true is not None:
        plt.scatter(X_axis, y_true, color="steelblue", s=20, alpha=0.6, label="Actual")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Predictive Uncertainty")
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = ["plot_actual_vs_predicted", "plot_uncertainty"]
