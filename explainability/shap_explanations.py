"""
SHAP values for local and global explainability.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _get_predict_fn(model, task="classification"):
    """Return predict function for SHAP (proba for classification)."""
    if task == "classification" and hasattr(model, "predict_proba"):
        return lambda x: model.predict_proba(x)[:, 1]
    return model.predict


def compute_shap_values(model, X_background, X_explain=None, task="classification", max_background=100):
    """
    Compute SHAP values. X_background: background set, X_explain: samples to explain.
    Returns: (shap_values, explainer).
    """
    try:
        import shap
    except ImportError:
        return None, None

    if X_explain is None:
        X_explain = X_background
    if hasattr(X_background, "values"):
        X_background = X_background.values
    if hasattr(X_explain, "values"):
        X_explain = X_explain.values

    n_bg = min(len(X_background), max_background or config.SHAP_SAMPLES)
    bg = X_background[np.random.RandomState(config.RANDOM_STATE).choice(len(X_background), n_bg, replace=False)]

    predict_fn = _get_predict_fn(model, task)
    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(predict_fn, masker)
    shap_vals = explainer(X_explain, silent=True)
    return shap_vals, explainer


def shap_summary_plot(shap_values, X, feature_names=None, save_path=None):
    """Plot SHAP summary (beeswarm or bar). Returns matplotlib figure."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    if hasattr(X, "values"):
        X = X.values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
