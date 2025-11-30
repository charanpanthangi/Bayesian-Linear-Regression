"""Run the full Bayesian linear regression workflow."""
from pathlib import Path

import numpy as np

from app.data import load_diabetes_data
from app.evaluate import regression_metrics
from app.model import BayesianLinearRegression
from app.preprocess import train_test_split_and_scale
from app.visualize import plot_actual_vs_predicted, plot_uncertainty


def summarize_uncertainty(std: np.ndarray) -> str:
    """Provide a simple textual summary of predictive uncertainty."""
    return (
        f"Average predictive std: {std.mean():.2f}\n"
        f"Min predictive std: {std.min():.2f}\n"
        f"Max predictive std: {std.max():.2f}"
    )


def main() -> None:
    # 1. Load data
    X, y = load_diabetes_data()

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler = train_test_split_and_scale(X, y)

    # 3. Train model
    model = BayesianLinearRegression()
    model.fit(X_train, y_train)

    # 4. Predict with uncertainty
    y_pred_mean, y_pred_std = model.predict(X_test)

    # 5. Evaluate
    metrics = regression_metrics(y_test, y_pred_mean)

    # 6. Visualize
    examples_dir = Path("examples")
    plot_actual_vs_predicted(
        y_test, y_pred_mean, examples_dir / "actual_vs_predicted.svg"
    )
    plot_uncertainty(
        range(len(y_pred_mean)),
        y_pred_mean,
        y_pred_std,
        y_test,
        examples_dir / "predictive_uncertainty.svg",
        xlabel="Test sample",
        ylabel="Disease progression",
    )

    # 7. Print results
    print("Bayesian Linear Regression on Diabetes dataset")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    print("\nUncertainty summary:")
    print(summarize_uncertainty(y_pred_std))


if __name__ == "__main__":
    main()
