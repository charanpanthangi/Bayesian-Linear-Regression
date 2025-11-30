from app.data import load_diabetes_data


def test_load_diabetes_data_shape():
    X, y = load_diabetes_data()
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
