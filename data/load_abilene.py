"""Generate synthetic Abilene-like traffic data."""

import numpy as np
import pandas as pd
import networkx as nx
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_abilene_graph():
    """Build Abilene network topology."""
    G = nx.Graph()
    for i in range(config.ABILENE_NODES):
        G.add_node(i)
    for u, v in config.ABILENE_EDGES:
        G.add_edge(u, v)
    return G


def generate_synthetic_traffic(G, n_timesteps=2000, seed=42):
    """
    Generate realistic traffic patterns.
    
    Returns:
        traffic_matrices: (n_timesteps, n_edges) array
        failure_labels: (n_timesteps,) binary array
    """
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    n_edges = len(edges)
    
    # Time-varying traffic
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    base_traffic = 50 + 30 * np.sin(t) + 20 * np.sin(0.5 * t)
    
    traffic_matrices = np.zeros((n_timesteps, n_edges))
    
    for edge_idx in range(n_edges):
        # Each edge has base + noise + edge-specific pattern
        edge_variation = rng.uniform(0.8, 1.2)
        traffic_matrices[:, edge_idx] = base_traffic * edge_variation + rng.normal(0, 10, n_timesteps)
        traffic_matrices[:, edge_idx] = np.clip(traffic_matrices[:, edge_idx], 1, 150)
    
    # Simulate failures based on traffic volatility
    failure_labels = np.zeros(n_timesteps, dtype=int)
    
    for i in range(1, n_timesteps):
        # High volatility → higher chance of prediction failure
        volatility = np.mean(np.abs(traffic_matrices[i] - traffic_matrices[i-1]))
        
        # Prediction error simulation (naive persistence model)
        pred = traffic_matrices[i-1]
        actual = traffic_matrices[i]
        rel_error = np.mean(np.abs(actual - pred) / (np.abs(actual) + 1e-6))
        
        # Add volatility-based bias
        if volatility > np.percentile(np.abs(np.diff(traffic_matrices, axis=0)), 75):
            rel_error *= 1.5  # Amplify error during volatile periods
        
        # Binary failure label
        if rel_error > config.FAILURE_ERROR_THRESHOLD:
            failure_labels[i] = 1
    
    return traffic_matrices, failure_labels


def simulate_topology_failures(G, traffic_matrices, failure_rate=0.2, seed=42):
    """
    Simulate random link failures and perturb traffic.
    
    Returns:
        G_failed: Graph with removed edges
        traffic_perturbed: Modified traffic
        failed_edges: List of (u, v) edges that failed
    """
    rng = np.random.default_rng(seed)
    G_failed = G.copy()
    edges = list(G.edges())
    
    n_failures = int(len(edges) * failure_rate)
    if n_failures == 0:
        return G_failed, traffic_matrices.copy(), []
    
    failed_edge_indices = rng.choice(len(edges), n_failures, replace=False)
    failed_edges = [edges[i] for i in failed_edge_indices]
    
    # Remove edges from graph
    for edge in failed_edges:
        if G_failed.has_edge(*edge):
            G_failed.remove_edge(*edge)
    
    # Perturb traffic: reroute from failed edges to others
    traffic_perturbed = traffic_matrices.copy()
    
    for idx in failed_edge_indices:
        if idx < traffic_matrices.shape[1]:
            # Redistribute failed link's traffic to remaining links
            failed_traffic = traffic_matrices[:, idx]
            remaining_indices = [i for i in range(len(edges)) if i not in failed_edge_indices]
            
            if remaining_indices:
                for i in remaining_indices:
                    traffic_perturbed[:, i] += failed_traffic / len(remaining_indices) * 0.3
            
            # Set failed link traffic to ~0
            traffic_perturbed[:, idx] = rng.uniform(0, 5, len(traffic_matrices))
    
    return G_failed, traffic_perturbed, failed_edges


def load_data(n_timesteps=None, seed=None):
    """
    Main data loading function.
    
    Returns:
        G: NetworkX graph
        traffic_matrices: (n_samples, n_edges)
        labels: (n_samples,) binary failure labels
    """
    n_timesteps = n_timesteps or config.N_TIMESTAMPS
    seed = seed or config.RANDOM_STATE
    
    G = build_abilene_graph()
    traffic, labels = generate_synthetic_traffic(G, n_timesteps, seed)
    
    return G, traffic, labels