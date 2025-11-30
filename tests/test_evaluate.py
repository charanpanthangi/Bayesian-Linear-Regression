import numpy as np

from app.evaluate import regression_metrics


def test_regression_metrics_values():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    metrics = regression_metrics(y_true, y_pred)

    assert metrics["MSE"] == np.mean((y_true - y_pred) ** 2)
    assert metrics["MAE"] == np.mean(np.abs(y_true - y_pred))
    assert np.isclose(metrics["RMSE"], np.sqrt(metrics["MSE"]))
    assert -np.inf < metrics["R2"] <= 1
