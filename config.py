"""Global configuration for CFA project."""

import numpy as np

# Random seed
RANDOM_STATE = 42

# Network topology (Abilene: 12 nodes)
ABILENE_NODES = 12
ABILENE_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (4, 5), (4, 7), (5, 6), (6, 8), (7, 9),
    (8, 9), (8, 10), (9, 11), (10, 11)
]

# Data paths
ABILENE_DATA_PATH = 'data/raw/abilene'

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Failure simulation
FAILURE_ERROR_THRESHOLD = 0.10  # 10% error threshold

# Failure rates for experiments
FAILURE_RATES = [0.1, 0.2, 0.3]

# GNN hyperparameters
GNN_HIDDEN_DIM = 128        # Increased
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.4           # Increased
GNN_LEARNING_RATE = 0.001
GNN_EPOCHS = 150            # Increased
GNN_BATCH_SIZE = 32

# CFA parameters
CFA_ERROR_THRESHOLD = 0.15
CFA_MAX_SEARCH_DEPTH = 2
CFA_TOP_K_CANDIDATES = 5

# Explainability
SHAP_SAMPLES = 100
RCA_TOP_K = 3

# Paths
DATA_DIR = "data/"
RESULTS_DIR = "results/"
FIGURES_DIR = "figures/"
MODELS_DIR = "models_saved/"