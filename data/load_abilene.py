"""Load and preprocess REAL Abilene network data ONLY."""

import numpy as np
import pandas as pd
import networkx as nx
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_abilene_graph():
    """Build Abilene network topology from the Abilene file."""
    
    abilene_file = Path('data/raw/abilene/Abilene')
    
    if not abilene_file.exists():
        raise FileNotFoundError(f"Topology file not found: {abilene_file}")
    
    G = nx.Graph()
    
    # Read topology file
    with open(abilene_file, 'r') as f:
        lines = f.readlines()
    
    # First line has node and edge count
    first_line = lines[0].strip()
    node_num = int(first_line.split()[1])
    
    # Add nodes
    for i in range(node_num):
        G.add_node(i)
    
    # Parse edges (skip header line 1)
    edges_seen = set()
    
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                source = int(parts[1])
                dest = int(parts[2])
                
                # Only add each edge once (undirected)
                edge = tuple(sorted([source, dest]))
                if edge not in edges_seen:
                    G.add_edge(source, dest)
                    edges_seen.add(edge)
            except:
                continue
    
    print(f"  Loaded topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def load_abilene_traffic_matrices(tm_file='data/raw/abilene/AbileneTM'):
    """
    Load Abilene traffic matrices from text file.
    
    Format: Each line = one 12×12 matrix flattened (144 values)
    
    Returns:
        od_matrices: (n_timesteps, 12, 12) numpy array
    """
    
    print(f"  Loading traffic matrices from {tm_file}...")
    
    tm_path = Path(tm_file)
    
    if not tm_path.exists():
        raise FileNotFoundError(f"Traffic matrix file not found: {tm_file}")
    
    # Read all lines
    with open(tm_path, 'r') as f:
        lines = f.readlines()
    
    print(f"  Found {len(lines)} timesteps")
    
    # Parse each line as a flattened matrix
    matrices = []
    
    for line in lines:
        values = line.strip().split()
        
        if len(values) != 144:
            print(f"  Warning: Line has {len(values)} values (expected 144), skipping")
            continue
        
        # Convert to floats
        try:
            row_values = [float(v) for v in values]
            
            # Reshape to 12×12
            matrix = np.array(row_values).reshape(12, 12)
            matrices.append(matrix)
            
        except ValueError as e:
            print(f"  Warning: Could not parse line: {e}")
            continue
    
    od_matrices = np.array(matrices)
    
    print(f"  Loaded {len(od_matrices)} traffic matrices")
    print(f"  Shape: {od_matrices.shape}")
    print(f"  Value range: {od_matrices.min():.2e} to {od_matrices.max():.2e}")
    
    # Convert from bytes to Mbps (assuming 5-minute intervals)
    # Bytes / (5 * 60 seconds) * 8 bits/byte / 1e6 = Mbps
    od_matrices_mbps = od_matrices / (5 * 60) * 8 / 1e6
    
    print(f"  Converted to Mbps: {od_matrices_mbps.min():.2f} to {od_matrices_mbps.max():.2f}")
    
    return od_matrices_mbps


def load_shortest_paths(sp_file='data/raw/abilene/Abilene_shortest_paths'):
    """
    Load pre-computed shortest paths.
    
    Returns:
        paths: Dict {(src, dst): [path_as_list]}
    """
    
    sp_path = Path(sp_file)
    
    if not sp_path.exists():
        print(f"  Warning: {sp_file} not found, will compute paths dynamically")
        return None
    
    paths = {}
    
    with open(sp_path, 'r') as f:
        for line in f:
            if '->' not in line:
                continue
            
            try:
                # Parse source and destination
                parts = line.strip().split(':')
                src_dst = parts[0].split('->')
                src = int(src_dst[0])
                dst = int(src_dst[1])
                
                # Parse path
                path_str = parts[1].strip()
                path_str = path_str.replace('[[', '[').replace(']]', ']')
                path = eval(path_str)
                
                paths[(src, dst)] = path
                
            except Exception as e:
                continue
    
    print(f"  Loaded {len(paths)} pre-computed paths")
    
    return paths


def od_matrix_to_edge_traffic(od_matrices, graph, shortest_paths=None):
    """
    Convert Origin-Destination matrices to per-edge traffic.
    
    Args:
        od_matrices: (n_timesteps, 12, 12) OD flow matrices
        graph: NetworkX graph
        shortest_paths: Optional pre-computed paths dict
    
    Returns:
        edge_traffic: (n_timesteps, n_edges) edge traffic array
    """
    
    print(f"  Converting OD matrices to edge traffic...")
    
    edges = list(graph.edges())
    n_edges = len(edges)
    n_timesteps = len(od_matrices)
    
    edge_traffic = np.zeros((n_timesteps, n_edges))
    
    # For each timestep
    for t in range(n_timesteps):
        od_matrix = od_matrices[t]
        
        # For each OD pair
        for src in range(12):
            for dst in range(12):
                if src == dst:
                    continue
                
                flow = od_matrix[src, dst]
                
                # Get shortest path
                if shortest_paths and (src, dst) in shortest_paths:
                    path = shortest_paths[(src, dst)]
                else:
                    try:
                        path = nx.shortest_path(graph, source=src, target=dst)
                    except nx.NetworkXNoPath:
                        continue
                
                # Add flow to edges on path
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    
                    # Find edge index (handle undirected)
                    if edge in edges:
                        edge_idx = edges.index(edge)
                    else:
                        edge = (path[i+1], path[i])
                        if edge in edges:
                            edge_idx = edges.index(edge)
                        else:
                            continue
                    
                    edge_traffic[t, edge_idx] += flow
        
        # Progress indicator
        if (t + 1) % 500 == 0:
            print(f"    Processed {t+1}/{n_timesteps} timesteps")
    
    print(f"  Edge traffic shape: {edge_traffic.shape}")
    print(f"  Edge traffic range: {edge_traffic.min():.2f} to {edge_traffic.max():.2f} Mbps")
    
    return edge_traffic


def generate_failure_labels(traffic_matrices, threshold=None):
    """
    Generate failure labels based on traffic prediction errors + volatility.
    
    Uses persistence model with volatility amplification.
    """
    
    threshold = threshold or config.FAILURE_ERROR_THRESHOLD
    
    n_timesteps = len(traffic_matrices)
    labels = np.zeros(n_timesteps, dtype=int)
    
    # Compute volatility for all timesteps
    volatility = np.zeros(n_timesteps)
    for t in range(1, n_timesteps):
        volatility[t] = np.mean(np.abs(traffic_matrices[t] - traffic_matrices[t-1]))
    
    volatility_threshold = np.percentile(volatility, 75)  # Top 25% volatility
    
    for t in range(1, n_timesteps):
        pred = traffic_matrices[t-1]
        actual = traffic_matrices[t]
        
        rel_error = np.mean(np.abs(actual - pred) / (np.abs(actual) + 1e-6))
        
        # Amplify error during high volatility periods
        if volatility[t] > volatility_threshold:
            rel_error *= 1.5
        
        if rel_error > threshold:
            labels[t] = 1
    
    failure_count = np.sum(labels)
    failure_rate = np.mean(labels)
    
    print(f"  Generated labels: {failure_count} failures ({failure_rate*100:.1f}%)")
    
    return labels


def simulate_topology_failures(G, traffic_matrices, failure_rate=0.2, seed=42):
    """
    Simulate random link failures and perturb traffic.
    """
    
    rng = np.random.default_rng(seed)
    G_failed = G.copy()
    edges = list(G.edges())
    
    n_failures = int(len(edges) * failure_rate)
    if n_failures == 0:
        return G_failed, traffic_matrices.copy(), []
    
    failed_edge_indices = rng.choice(len(edges), n_failures, replace=False)
    failed_edges = [edges[i] for i in failed_edge_indices]
    
    # Remove edges
    for edge in failed_edges:
        if G_failed.has_edge(*edge):
            G_failed.remove_edge(*edge)
    
    # Perturb traffic
    traffic_perturbed = traffic_matrices.copy()
    
    for idx in failed_edge_indices:
        if idx < traffic_matrices.shape[1]:
            failed_traffic = traffic_matrices[:, idx]
            remaining_indices = [i for i in range(len(edges)) if i not in failed_edge_indices]
            
            if remaining_indices:
                for i in remaining_indices:
                    traffic_perturbed[:, i] += failed_traffic / len(remaining_indices) * 0.3
            
            traffic_perturbed[:, idx] = rng.uniform(0, 5, len(traffic_matrices))
    
    return G_failed, traffic_perturbed, failed_edges


def load_data():
    """
    Load REAL Abilene data ONLY.
    
    Returns:
        G: NetworkX graph
        traffic_matrices: (n_samples, n_edges)
        labels: (n_samples,) binary failure labels
    """
    
    print("\n" + "="*80)
    print("LOADING REAL ABILENE DATA")
    print("="*80)
    
    # Step 1: Load topology
    G = build_abilene_graph()
    
    # Step 2: Load traffic matrices (OD format)
    od_matrices = load_abilene_traffic_matrices()
    
    # Step 3: Load shortest paths (optional)
    shortest_paths = load_shortest_paths()
    
    # Step 4: Convert OD matrices to edge traffic
    edge_traffic = od_matrix_to_edge_traffic(od_matrices, G, shortest_paths)
    
    # Step 5: Generate failure labels
    labels = generate_failure_labels(edge_traffic)
    
    print("\n" + "="*80)
    print("REAL ABILENE DATA LOADED SUCCESSFULLY")
    print("="*80)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Timesteps: {len(edge_traffic)}")
    print(f"  Failures: {np.sum(labels)} ({100*np.mean(labels):.1f}%)")
    print("="*80 + "\n")
    
    return G, edge_traffic, labels