"""
Feature importance: model-specific (tree/RF coefficients) and model-agnostic (permutation).
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_model_importance(model, feature_names):
    """
    Model-specific importance: tree feature_importances_ or logistic coefficients.
    Returns: dict mapping feature_name -> importance (absolute for linear).
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
        if imp.shape[0] != len(feature_names):
            imp = np.abs(model.coef_[0])
    else:
        return {}
    return dict(zip(feature_names, imp))


def get_permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="accuracy"):
    """
    Model-agnostic permutation importance.
    Returns: DataFrame with columns feature, importance_mean, importance_std.
    """
    if hasattr(model, "predict_proba"):
        scorer = scoring  # e.g. 'accuracy', 'f1'
    else:
        scorer = "neg_mean_squared_error" if scoring in ("accuracy", "f1") else scoring
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scorer
    )
    df = pd.DataFrame({
        "feature": X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False)
