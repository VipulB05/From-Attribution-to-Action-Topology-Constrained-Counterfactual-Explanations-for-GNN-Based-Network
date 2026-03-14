"""
Load and preprocess network traffic data for prediction failure analysis.
Supports Abilene-like synthetic data and CSV input (e.g., from SNDlib exports).
Target: prediction failure (binary) or prediction error (regression).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import networkx as nx
import sys

# Add project root for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _build_abilene_graph():
    """Build NetworkX graph for Abilene-like topology (12 nodes)."""
    G = nx.Graph()
    for i in range(config.ABILENE_NODES):
        G.add_node(i)
    for u, v in config.ABILENE_EDGES:
        G.add_edge(u, v)
    return G


def _compute_topology_features(G):
    """Compute per-node topology features (degree, betweenness, etc.)."""
    degree = dict(nx.degree(G))
    betweenness = nx.betweenness_centrality(G)
    return {"degree": degree, "betweenness": betweenness}


def generate_synthetic_abilene_traffic(n_timesteps=None, seed=None):
    """
    Generate synthetic link-level traffic consistent with Abilene-like topology.
    Returns DataFrame with columns: time_idx, link_0, link_1, ... (one per edge),
    and node-level features derived from topology.
    """
    n_timesteps = n_timesteps or config.N_TIMESTAMPS
    rng = np.random.default_rng(seed or config.RANDOM_STATE)
    G = _build_abilene_graph()
    edges = list(G.edges())
    n_links = len(edges)

    # Link traffic: smooth + noise (realistic for network traffic)
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    base = 50 + 30 * np.sin(t) + 20 * np.sin(0.5 * t)
    link_traffic = np.zeros((n_timesteps, n_links))
    for j in range(n_links):
        link_traffic[:, j] = base + rng.normal(0, 10, n_timesteps)
        link_traffic[:, j] = np.clip(link_traffic[:, j], 1, 150)

    # Node features from topology
    topo = _compute_topology_features(G)
    n_nodes = G.number_of_nodes()
    degree_feat = np.array([topo["degree"][i] for i in range(n_nodes)])
    betweenness_feat = np.array([topo["betweenness"][i] for i in range(n_nodes)])

    # Build DataFrame: time, link loads, then repeated node features per timestep
    cols = ["time_idx"] + [f"link_{i}" for i in range(n_links)]
    df = pd.DataFrame(np.c_[np.arange(n_timesteps), link_traffic], columns=cols)

    # Add node degree/betweenness (repeated for each timestep - same topology)
    for i in range(n_nodes):
        df[f"node{i}_degree"] = degree_feat[i]
        df[f"node{i}_betweenness"] = betweenness_feat[i]

    # Lagged features (past traffic) for prediction
    for j in range(min(3, n_links)):
        df[f"link_{j}_lag1"] = df[f"link_{j}"].shift(1).bfill()
        df[f"link_{j}_lag2"] = df[f"link_{j}"].shift(2).bfill()
    df = df.dropna().reset_index(drop=True)

    return df, G, edges


def simulate_prediction_and_failure(df, target_col_pattern="link_", threshold=None):
    """
    Simulate a simple predictor (e.g., persistence) and compute per-row prediction error.
    Adds columns: pred_error (relative), failure (binary).
    """
    threshold = threshold or config.FAILURE_ERROR_THRESHOLD
    link_cols = [c for c in df.columns if c.startswith(target_col_pattern) and "_lag" not in c and c != "time_idx"]
    if not link_cols:
        link_cols = [c for c in df.columns if "link_" in c and "_lag" not in c]

    # Simulate prediction: naive persistence (prev value) + some failure-inducing conditions
    df = df.copy()
    errors = []
    for i in range(len(df)):
        if i == 0:
            err = 0.0
        else:
            # "Predicted" = previous + bias that increases under high load (simulating GNN failure modes)
            pred = df[link_cols].iloc[i - 1].values
            actual = df[link_cols].iloc[i].values
            # Introduce higher error when traffic is volatile (simulate failure)
            volatility = np.abs(actual - pred)
            bias = np.where(volatility > np.median(volatility), 0.15, -0.05)
            pred = pred * (1 + bias)
            rel_err = np.mean(np.abs(actual - pred) / (np.abs(actual) + 1e-6))
            err = rel_err
        errors.append(err)

    df["pred_error"] = errors
    df["failure"] = (df["pred_error"] >= threshold).astype(int)
    return df


def load_abilene_like_data(data_path=None, n_timesteps=None, use_synthetic=True):
    """
    Load Abilene-like data. If data_path is provided and exists, load from CSV;
    otherwise generate synthetic.
    Returns: (df_with_targets, graph, feature_names, target_name).
    """
    if data_path and os.path.isfile(data_path):
        df = pd.read_csv(data_path)
        G = _build_abilene_graph()
        edges = list(G.edges())
        # Assume CSV has link_* columns; if not, try to infer
        if "failure" not in df.columns and "pred_error" not in df.columns:
            df = simulate_prediction_and_failure(df)
        feature_cols = [c for c in df.columns if c not in ("failure", "pred_error", "time_idx")]
        return df, G, feature_cols, "failure"
    else:
        df, G, edges = generate_synthetic_abilene_traffic(n_timesteps=n_timesteps)
        df = simulate_prediction_and_failure(df)
        feature_cols = [c for c in df.columns if c not in ("failure", "pred_error", "time_idx")]
        return df, G, feature_cols, "failure"


def preprocess_dataset(df, feature_cols, target_name, impute=True, normalize=True, scaler=None):
    """
    Preprocess: impute missing values, optionally normalize.
    Returns: X (array), y (array), feature_names, fitted scaler (or None).
    """
    X = df[feature_cols].copy()
    y = df[target_name].values

    if impute:
        imp = SimpleImputer(strategy="median")
        X = pd.DataFrame(imp.fit_transform(X), columns=feature_cols)

    if normalize:
        scaler = scaler or StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    else:
        scaler = None

    return X, y, list(X.columns), scaler


def get_train_test_split(X, y, train_ratio=None, stratify=None, random_state=None):
    """Train/test split with optional stratification (for binary failure)."""
    train_ratio = train_ratio or config.TRAIN_RATIO
    random_state = random_state or config.RANDOM_STATE
    stratify = stratify if stratify is not None else y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=stratify, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
