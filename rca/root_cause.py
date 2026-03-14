"""
Root cause analysis: identify key drivers of target (failure), causal inference (DoWhy), sensitivity.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def identify_key_drivers(feature_importance_dict, top_k=10):
    """
    From feature importance (model or permutation), return top drivers.
    Returns: list of (feature_name, importance).
    """
    if isinstance(feature_importance_dict, dict):
        items = sorted(feature_importance_dict.items(), key=lambda x: -abs(x[1]))[:top_k]
    else:
        items = list(feature_importance_dict.items())[:top_k]
    return items


def run_causal_analysis(df, treatment_cols, outcome_col, common_causes=None):
    """
    Run DoWhy causal analysis: effect of treatment_cols on outcome_col.
    Returns: causal estimate and summary.
    """
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        return None, "DoWhy not installed; pip install dowhy"

    if common_causes is None:
        common_causes = [c for c in df.columns if c not in treatment_cols and c != outcome_col][:5]

    model = CausalModel(
        data=df,
        treatment=treatment_cols if isinstance(treatment_cols, list) else [treatment_cols],
        outcome=outcome_col,
        common_causes=common_causes,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    try:
        estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    except TypeError:
        # Older DoWhy API
        estimate = model.estimate_effect(method_name="backdoor.linear_regression")
    return estimate, model


def sensitivity_analysis(model, X, y, feature_names, n_perturbations=20, noise_scale=0.1):
    """
    Sensitivity: perturb features and measure change in prediction.
    Returns: dict of feature -> mean absolute change in prediction.
    """
    rng = np.random.RandomState(42)
    X_arr = X.values if hasattr(X, "values") else np.array(X)
    preds_base = model.predict(X_arr)
    if hasattr(model, "predict_proba"):
        preds_base = model.predict_proba(X_arr)[:, 1]

    sensitivity = {}
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_perturbations):
            X_pert = X_arr.copy()
            X_pert[:, j] += rng.normal(0, noise_scale * (X_pert[:, j].std() + 1e-6), X_pert.shape[0])
            if hasattr(model, "predict_proba"):
                preds_pert = model.predict_proba(X_pert)[:, 1]
            else:
                preds_pert = model.predict(X_pert)
            deltas.append(np.mean(np.abs(preds_pert - preds_base)))
        sensitivity[name] = np.mean(deltas)
    return sensitivity
