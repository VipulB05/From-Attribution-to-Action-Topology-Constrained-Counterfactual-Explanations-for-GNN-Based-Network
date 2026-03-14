"""Explainability: feature importance, SHAP, LIME, counterfactuals."""

from .feature_importance import get_model_importance, get_permutation_importance
from .shap_explanations import compute_shap_values, shap_summary_plot
from .lime_explanations import get_lime_explanations
from .counterfactuals import generate_counterfactuals

__all__ = [
    "get_model_importance",
    "get_permutation_importance",
    "compute_shap_values",
    "shap_summary_plot",
    "get_lime_explanations",
    "generate_counterfactuals",
]
