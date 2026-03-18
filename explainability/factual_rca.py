"""
Factual Root Cause Analysis
Identifies which components contributed to prediction failure.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
from torch_geometric.data import Data


class _ExplainerModelWrapper(torch.nn.Module):
    """Adapter so PyG Explainer can call models that expect a Data object."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.base_model(data)


class FactualRCA:
    """
    Factual explanation: "What caused this failure?"
    Uses GNNExplainer + topology features.
    """
    
    def __init__(self, model, graph, device='cpu'):
        self.model = model
        self.graph = graph
        self.device = device
        self.explainer_model = _ExplainerModelWrapper(model).to(device)
        
        # GNNExplainer for model attribution
        self.explainer = Explainer(
            model=self.explainer_model,
            algorithm=PyGGNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='log_probs',
            ),
        )
        
        # Precompute topology features
        self.topology_features = self._compute_topology_features()
    
    def _compute_topology_features(self):
        """Compute structural importance of each component."""
        features = {}
        
        # Node-level metrics
        degree = dict(self.graph.degree())
        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)
        
        for node in self.graph.nodes():
            features[f'node_{node}'] = {
                'type': 'node',
                'id': node,
                'degree': degree[node],
                'betweenness': betweenness[node],
                'closeness': closeness[node],
                'structural_risk': self._compute_node_risk(
                    degree[node], betweenness[node], closeness[node]
                )
            }
        
        # Edge-level metrics
        edge_betweenness = nx.edge_betweenness_centrality(self.graph)
        
        for edge in self.graph.edges():
            u, v = edge
            features[f'edge_{u}_{v}'] = {
                'type': 'edge',
                'id': edge,
                'betweenness': edge_betweenness[edge],
                'connects_high_degree': degree[u] > 3 or degree[v] > 3,
                'structural_risk': edge_betweenness[edge]
            }
        
        return features
    
    def _compute_node_risk(self, degree, betweenness, closeness):
        """
        Compute structural risk score for a node.
        High risk = high betweenness (bottleneck) + low degree (few alternatives)
        """
        # Normalize to [0, 1]
        # High betweenness = risky, low degree = risky
        risk = 0.6 * betweenness + 0.3 * (1.0 / (degree + 1)) + 0.1 * closeness
        return risk
    
    def explain(self, data, top_k=3):
        """
        Generate factual explanation for a failed prediction.
        
        Args:
            data: PyG Data object with failed prediction
            top_k: Number of root causes to return
        
        Returns:
            List of root causes with importance scores
        """
        data = data.to(self.device)

        # Get model attribution
        explanation = self.explainer(data.x, data.edge_index, batch=data.batch)
        
        # Node importance from GNNExplainer
        node_mask = explanation.node_mask.cpu().numpy()
        
        # Edge importance (if available)
        edge_mask = explanation.edge_mask.cpu().numpy() if hasattr(explanation, 'edge_mask') else np.zeros(data.edge_index.shape[1])
        
        # Combine with topology features
        combined_scores = {}
        
        # Score nodes
        for node_id in range(len(node_mask)):
            if node_id < len(self.topology_features):
                key = f'node_{node_id}'
                if key in self.topology_features:
                    model_importance = float(np.mean(node_mask[node_id]))
                    topo_risk = self.topology_features[key]['structural_risk']
                    
                    # Weighted combination
                    combined = 0.6 * model_importance + 0.4 * topo_risk
                    
                    combined_scores[key] = {
                        'type': 'node',
                        'id': node_id,
                        'score': combined,
                        'model_attribution': float(model_importance),
                        'topology_risk': float(topo_risk),
                        'details': self.topology_features[key]
                    }
        
        # Sort by score
        ranked = sorted(combined_scores.values(), key=lambda x: x['score'], reverse=True)
        
        return ranked[:top_k]
    
    def batch_explain(self, data_list):
        """Explain multiple failures."""
        results = []
        for data in data_list:
            explanation = self.explain(data)
            results.append(explanation)
        return results