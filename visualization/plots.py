"""
Plotting: confusion matrix, ROC, feature importance bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Plot ROC curve if binary classification."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(importance_dict, top_k=15, save_path=None, title="Feature Importance"):
    """Bar plot of feature importance."""
    items = sorted(importance_dict.items(), key=lambda x: -abs(x[1]))[:top_k]
    names = [x[0][:30] for x in items]
    vals = [x[1] for x in items]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
