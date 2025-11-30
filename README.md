# Bayesian Linear Regression with Predictive Uncertainty

A beginner-friendly, production-ready template demonstrating **Bayesian Linear Regression** with uncertainty estimation on the scikit-learn diabetes dataset. The project mixes tutorial explanations with modular code so you can learn the concepts and immediately reuse the components.

## Why Bayesian Linear Regression?
Bayesian Linear Regression treats model parameters as probability distributions rather than fixed values. Compared to deterministic linear regression, the Bayesian approach:

- Captures parameter uncertainty, producing predictive mean **and** variance.
- Provides credible intervals that shrink or grow with available data, making it robust in low-data settings.
- Supports principled regularization via priors, reducing overfitting.

The posterior predictive distribution gives:

- **Predictive mean**: the expected target value for new data.
- **Predictive variance**: how uncertain the model is about that prediction (larger variance = less confidence).

## Project Structure
```
├── app/
│   ├── data.py          # Load diabetes dataset
│   ├── preprocess.py    # Train/test split and scaling
│   ├── model.py         # Bayesian Ridge model wrapper with uncertainty
│   ├── evaluate.py      # Regression metrics (MSE, MAE, RMSE, R²)
│   ├── visualize.py     # Actual vs predicted and uncertainty plots
│   ├── main.py          # End-to-end pipeline
├── notebooks/
│   └── demo_bayesian_linear_regression.ipynb
├── tests/               # Pytest unit tests
├── examples/            # Generated plots live here
├── requirements.txt
├── Dockerfile
├── LICENSE (MIT)
└── README.md
```

## Dataset
The [scikit-learn diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) contains 442 samples with 10 baseline variables (e.g., age, BMI, blood pressure) and a continuous target measuring disease progression one year after baseline.

## Pipeline Overview
1. **Load** the diabetes dataset (`app/data.py`).
2. **Split & scale** features using `StandardScaler` (`app/preprocess.py`).
3. **Train** a Bayesian ridge regression model (`app/model.py`).
4. **Predict** the posterior predictive mean and standard deviation.
5. **Evaluate** using MSE, MAE, RMSE, and R² (`app/evaluate.py`).
6. **Visualize** actual vs. predicted values and uncertainty bands (`app/visualize.py`).

## Getting Started
### Option 1: Local environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

### Option 2: Docker
```bash
docker build -t bayesian-linear-regression .
docker run --rm bayesian-linear-regression
```

### Option 3: Notebook
Launch Jupyter and open `notebooks/demo_bayesian_linear_regression.ipynb`:
```bash
jupyter notebook
```

## Outputs
Running `python app/main.py` generates SVG plots in `examples/`:
- `actual_vs_predicted.svg`
- `predictive_uncertainty.svg`

The console prints regression metrics and a summary of predictive uncertainty (average/min/max standard deviation).

## Future Extensions
- Swap in `ARDRegression` to explore automatic relevance determination.
- Tune priors (`alpha_1`, `lambda_1`, etc.) to encode domain knowledge.
- Compare against standard Linear Regression or Ridge to see how uncertainty changes.
- Log runs with MLflow or Weights & Biases.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
