"""
Evaluation: accuracy, F1, RMSE; interpretability: explanation stability, consistency.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix,
)


def compute_standard_metrics(y_true, y_pred, y_proba=None, task="classification"):
    """
    Standard metrics. For classification: accuracy, F1, precision, recall, AUC.
    For regression: RMSE, MAE.
    """
    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                proba_pos = y_proba[:, 1] if np.asarray(y_proba).ndim > 1 else np.asarray(y_proba).ravel()
                metrics["roc_auc"] = roc_auc_score(y_true, proba_pos)
            except Exception:
                metrics["roc_auc"] = 0.0
        return metrics
    else:
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": np.mean(np.abs(np.array(y_true) - np.array(y_pred))),
        }


def explanation_stability(shap_values_list, feature_names):
    """
    Stability of explanations across samples: variance of feature rankings or mean absolute SHAP.
    shap_values_list: list of arrays (one per sample) or single (n_samples, n_features).
    Returns: per-feature std of absolute SHAP (lower = more stable).
    """
    if hasattr(shap_values_list, "values"):
        vals = np.abs(shap_values_list.values)
    else:
        vals = np.abs(np.array(shap_values_list))
    if vals.ndim == 2:
        return dict(zip(feature_names, np.std(vals, axis=0)))
    return {f: np.std(vals) for f in feature_names}


def explanation_consistency(importance_1, importance_2, feature_names):
    """
    Consistency between two importance vectors (e.g. model vs permutation).
    Spearman correlation or top-k overlap.
    """
    if not feature_names:
        return 0.0
    v1 = np.array([importance_1.get(f, 0) for f in feature_names])
    v2 = np.array([importance_2.get(f, 0) for f in feature_names])
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    from scipy.stats import spearmanr
    r, _ = spearmanr(v1, v2)
    return float(r) if not np.isnan(r) else 0.0


def compute_interpretability_metrics(
    model_importance,
    perm_importance_df,
    shap_values=None,
    X=None,
    feature_names=None,
):
    """
    Interpretability metrics: stability (if SHAP provided), consistency (model vs permutation).
    Returns dict.
    """
    if feature_names is None and perm_importance_df is not None:
        feature_names = list(perm_importance_df["feature"])
    elif feature_names is None:
        feature_names = list(model_importance.keys()) if isinstance(model_importance, dict) else []

    perm_dict = {}
    if perm_importance_df is not None and not perm_importance_df.empty:
        perm_dict = dict(zip(perm_importance_df["feature"], perm_importance_df["importance_mean"]))

    consistency = explanation_consistency(model_importance, perm_dict, feature_names)

    out = {"importance_consistency": consistency}
    if shap_values is not None and X is not None and feature_names:
        # Handle SHAP Explanation object
        vals = getattr(shap_values, "values", shap_values)
        if hasattr(vals, "values"):
            vals = np.asarray(vals)
        stab = explanation_stability(vals, feature_names)
        out["explanation_stability_mean_std"] = float(np.mean(list(stab.values())))
    return out
