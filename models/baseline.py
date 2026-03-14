"""
Baseline models for failure prediction (binary) or error regression.
Supports: Logistic Regression, Random Forest, Gradient Boosting.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_baseline_model(
    X_train, y_train, task="classification", model_type=None, random_state=None
):
    """
    Train a baseline model. task: 'classification' (failure) or 'regression' (pred_error).
    model_type: 'logistic', 'random_forest', 'gradient_boosting'.
    Returns fitted model and task type.
    """
    model_type = model_type or config.BASELINE_MODEL
    random_state = random_state or config.RANDOM_STATE

    if task == "classification":
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type in ("random_forest", "rf"):
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type in ("gradient_boosting", "gbm"):
            model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        if model_type == "logistic":
            raise ValueError("Logistic regression is for classification only.")
        if model_type in ("random_forest", "rf"):
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)

    model.fit(X_train, y_train)
    return model, task


def get_baseline_predictions(model, X, task="classification"):
    """Return predictions (class labels or continuous)."""
    if task == "classification":
        return model.predict(X), model.predict_proba(X)
    return model.predict(X), None
