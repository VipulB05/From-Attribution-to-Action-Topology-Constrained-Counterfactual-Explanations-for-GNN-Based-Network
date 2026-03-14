"""
Counterfactual examples: minimal change to flip prediction (failure -> no failure).
Simple optimization-based counterfactuals without external lib.
"""

import numpy as np
import pandas as pd


def generate_counterfactuals(
    model, X_instance, feature_names, target_class=0, n_cf=5, step=0.1, max_steps=200
):
    """
    Generate counterfactuals by perturbing features until prediction flips to target_class.
    X_instance: one row (Series or array). target_class: desired class (e.g. 0 = no failure).
    Returns: list of (counterfactual_row, n_steps).
    """
    X = np.array(X_instance).reshape(1, -1)
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        proba = None

    results = []
    n_features = X.shape[1]
    rng = np.random.RandomState(42)

    for _ in range(n_cf):
        x = X.copy().ravel()
        for _ in range(max_steps):
            if model.predict(x.reshape(1, -1))[0] == target_class:
                results.append((x.copy(), _))
                break
            # Gradient-free: random direction
            delta = rng.randn(n_features) * step
            x_new = x + delta
            x_new = np.clip(x_new, x - 2, x + 2)  # bounded change
            if model.predict(x_new.reshape(1, -1))[0] == target_class:
                results.append((x_new, _ + 1))
                break
            # Move toward decision boundary: if we're predicting 1, try reducing features that push toward 1
            x = x_new

    if not results:
        results = [(X.ravel(), max_steps)]
    cf_df = pd.DataFrame(np.array([r[0] for r in results]), columns=feature_names)
    return cf_df, results
