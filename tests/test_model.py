import numpy as np

from app.model import BayesianLinearRegression


def test_model_fit_and_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    true_coefs = np.array([1.5, -2.0, 0.5])
    y = X @ true_coefs + rng.normal(scale=0.1, size=30)

    model = BayesianLinearRegression()
    model.fit(X, y)
    mean, std = model.predict(X[:5])

    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std > 0)
