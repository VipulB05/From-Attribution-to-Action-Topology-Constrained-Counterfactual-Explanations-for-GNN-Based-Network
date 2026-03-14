"""
LIME for local explanations (tabular).
"""

import numpy as np
import pandas as pd


def get_lime_explanations(model, X, instance_idx, feature_names=None, task="classification", num_samples=500):
    """
    Get LIME explanation for a single instance.
    Returns: (explanation object, list of (feature, weight) for the class of interest).
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return None, []

    X_arr = X.values if hasattr(X, "values") else np.array(X)
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    else:
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

    mode = "classification" if task == "classification" else "regression"
    predict_fn = model.predict_proba if (task == "classification" and hasattr(model, "predict_proba")) else model.predict
    explainer = LimeTabularExplainer(
        X_arr,
        feature_names=feature_names,
        mode=mode,
        random_state=42,
    )
    exp = explainer.explain_instance(
        X_arr[instance_idx],
        predict_fn,
        num_features=min(15, len(feature_names)),
        num_samples=num_samples,
    )
    # Top features for positive class (failure)
    if task == "classification":
        weights = exp.as_list(label=1) if hasattr(exp, "as_list") else exp.local_exp.get(1, [])
    else:
        weights = exp.as_list()
    return exp, weights
