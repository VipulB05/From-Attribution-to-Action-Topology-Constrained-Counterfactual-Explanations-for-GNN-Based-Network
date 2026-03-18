"""Baseline models: Logistic Regression, Random Forest, Gradient Boosting."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch_geometric.loader import DataLoader


def dataset_to_tabular(dataset):
    """Convert PyG dataset to tabular format for sklearn."""
    X = []
    y = []
    
    for data in dataset:
        # Flatten node features
        node_features = data.x.numpy().flatten()
        X.append(node_features)
        y.append(data.y.item())
    
    return np.array(X), np.array(y)


def train_baseline_models(train_dataset, val_dataset, test_dataset):
    """Train all baseline models and return results."""
    
    # Convert to tabular
    X_train, y_train = dataset_to_tabular(train_dataset)
    X_val, y_val = dataset_to_tabular(val_dataset)
    X_test, y_test = dataset_to_tabular(test_dataset)
    
    results = {}
    
    # Logistic Regression
    print("    Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    results['logistic'] = {
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'labels': y_test,
        'accuracy': accuracy_score(y_test, lr_pred),
        'f1': f1_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_proba)
    }
    
    # Random Forest
    print("    Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    results['random_forest'] = {
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'labels': y_test,
        'accuracy': accuracy_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_proba)
    }
    
    # Gradient Boosting
    print("    Training Gradient Boosting...")
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]
    
    results['gbm'] = {
        'predictions': gbm_pred,
        'probabilities': gbm_proba,
        'labels': y_test,
        'accuracy': accuracy_score(y_test, gbm_pred),
        'f1': f1_score(y_test, gbm_pred),
        'precision': precision_score(y_test, gbm_pred),
        'recall': recall_score(y_test, gbm_pred),
        'roc_auc': roc_auc_score(y_test, gbm_proba)
    }
    
    return results