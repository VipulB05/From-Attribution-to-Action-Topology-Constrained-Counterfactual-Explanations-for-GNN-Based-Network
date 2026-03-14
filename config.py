"""
Configuration for Explainable Root Cause Analysis pipeline.
Aligned with: Explainable RCA for GNN Traffic Prediction Failures (Abilene/GÉANT).
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Abilene-like network (12 nodes)
ABILENE_NODES = 12
# Simplified Abilene topology: node index -> list of neighbors (undirected)
ABILENE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # ring
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6),
]

# Data generation (when using synthetic)
N_TIMESTAMPS = 500   # time steps per run
TRAIN_RATIO = 0.75
RANDOM_STATE = 42

# Failure threshold: prediction is "failure" if relative error > this
FAILURE_ERROR_THRESHOLD = 0.25  # 25% relative error

# Model
BASELINE_MODEL = "random_forest"  # "random_forest" | "logistic" | "gradient_boosting"

# Explainability
SHAP_SAMPLES = 100   # max samples for SHAP (for speed)
LIME_SAMPLES = 500
N_COUNTERFACTUALS = 5

# RCA
CAUSAL_CONFOUNDERS = None  # will be set from feature names
