"""PyTorch Geometric Dataset for network traffic."""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class NetworkTrafficDataset(Dataset):
    """Dataset of graph snapshots with traffic and failures."""
    
    def __init__(self, graph: nx.Graph, traffic_matrices: np.ndarray, 
                 labels: np.ndarray, transform=None, pre_transform=None):
        """
        Args:
            graph: NetworkX graph (static topology)
            traffic_matrices: (n_samples, n_edges) array of edge traffic
            labels: (n_samples,) binary failure labels
        """
        super().__init__(None, transform, pre_transform)
        self.graph = graph
        self.traffic_matrices = traffic_matrices
        self.labels = labels
        
        # Convert NetworkX to PyG edge_index
        self.edge_index = self._get_edge_index()
        self.num_nodes = graph.number_of_nodes()
        
    def _get_edge_index(self):
        """Convert NetworkX edges to PyTorch Geometric format."""
        edge_list = list(self.graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Make undirected (add reverse edges)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index
    
    def _compute_node_features(self, edge_traffic: np.ndarray) -> torch.Tensor:
        """Aggregate edge traffic to node features."""
        # Node features: degree + aggregated incident edge traffic
        node_features = np.zeros((self.num_nodes, 3))
        
        # Degree (static)
        degree = dict(self.graph.degree())
        for i in range(self.num_nodes):
            node_features[i, 0] = degree[i]
        
        # Betweenness (static, precomputed)
        betweenness = nx.betweenness_centrality(self.graph)
        for i in range(self.num_nodes):
            node_features[i, 1] = betweenness[i]
        
        # Aggregated traffic (dynamic)
        edge_list = list(self.graph.edges())
        for edge_idx, (u, v) in enumerate(edge_list):
            if edge_idx < len(edge_traffic):
                traffic = edge_traffic[edge_idx]
                node_features[u, 2] += traffic
                node_features[v, 2] += traffic
        
        return torch.tensor(node_features, dtype=torch.float)
    
    def len(self):
        return len(self.labels)
    
    def get(self, idx):
        """Get graph data for one timestep."""
        # Node features
        x = self._compute_node_features(self.traffic_matrices[idx])
        
        # Edge features (traffic values)
        edge_traffic = self.traffic_matrices[idx]
        edge_attr = torch.tensor(edge_traffic, dtype=torch.float).unsqueeze(1)
        # Duplicate for undirected edges
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Label
        y = torch.tensor([self.labels[idx]], dtype=torch.long)
        
        data = Data(x=x, edge_index=self.edge_index, edge_attr=edge_attr, y=y)
        return data


def create_datasets(graph, traffic_matrices, labels, 
                   train_ratio=0.7, val_ratio=0.15, random_state=42):
    """Split into train/val/test datasets."""
    n = len(labels)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_dataset = NetworkTrafficDataset(graph, traffic_matrices[train_idx], labels[train_idx])
    val_dataset = NetworkTrafficDataset(graph, traffic_matrices[val_idx], labels[val_idx])
    test_dataset = NetworkTrafficDataset(graph, traffic_matrices[test_idx], labels[test_idx])
    
    return train_dataset, val_dataset, test_dataset