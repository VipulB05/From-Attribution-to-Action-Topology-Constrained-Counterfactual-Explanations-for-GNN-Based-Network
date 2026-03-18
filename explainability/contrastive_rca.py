"""
Contrastive Root Cause Analysis (CFA)
Finds minimal interventions to restore prediction accuracy.
"""

import torch
import numpy as np
import networkx as nx
from copy import deepcopy
from torch_geometric.data import Data
from itertools import combinations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ContrastiveRCA:
    """
    Counterfactual explanation: "What's the minimal fix?"
    
    Strategy:
    1. Identify failed components (edges/nodes)
    2. Try restoring them one-by-one
    3. Find minimal intervention that reduces error below threshold
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain(self, data, failed_graph, original_graph, 
                error_threshold=None, max_depth=None):
        """
        Generate counterfactual explanation.
        
        Args:
            data: Current PyG Data object (with failure)
            failed_graph: NetworkX graph after failures
            original_graph: NetworkX graph before failures
            error_threshold: Maximum acceptable error (default from config)
            max_depth: Maximum restoration size (default from config)
        
        Returns:
            Dict with minimal interventions ranked by effectiveness
        """
        error_threshold = error_threshold or config.CFA_ERROR_THRESHOLD
        max_depth = max_depth or config.CFA_MAX_SEARCH_DEPTH
        
        # Identify what failed
        failed_edges = list(set(original_graph.edges()) - set(failed_graph.edges()))
        failed_nodes = list(set(original_graph.nodes()) - set(failed_graph.nodes()))
        
        if not failed_edges and not failed_nodes:
            return {
                'success': False,
                'message': 'No failures detected',
                'interventions': []
            }
        
        # Get current prediction error
        current_error = self._compute_prediction_error(data)
        
        print(f"\n  [CFA] Current error: {current_error:.3f}")
        print(f"  [CFA] Failed edges: {len(failed_edges)}, Failed nodes: {len(failed_nodes)}")
        
        # Search for minimal intervention
        interventions = []
        
        # Level 1: Single component restorations
        print(f"  [CFA] Searching single-component interventions...")
        single_interventions = self._search_single_restorations(
            data, failed_graph, original_graph, 
            failed_edges, failed_nodes,
            error_threshold, current_error
        )
        interventions.extend(single_interventions)
        
        # Check if we found a solution
        successful = [i for i in interventions if i['success']]
        if successful:
            print(f"  [CFA] ✓ Found {len(successful)} successful single interventions")
            return self._format_results(interventions, current_error)
        
        # Level 2: Pairs (only if max_depth >= 2)
        if max_depth >= 2 and len(failed_edges) > 1:
            print(f"  [CFA] Searching pair interventions...")
            pair_interventions = self._search_pair_restorations(
                data, failed_graph, original_graph,
                failed_edges, failed_nodes,
                error_threshold, current_error
            )
            interventions.extend(pair_interventions)
        
        return self._format_results(interventions, current_error)
    
    def _search_single_restorations(self, data, failed_graph, original_graph,
                                    failed_edges, failed_nodes, 
                                    error_threshold, current_error):
        """Try restoring single edges/nodes."""
        interventions = []
        
        # Try each failed edge
        for edge in failed_edges:
            G_restored = failed_graph.copy()
            G_restored.add_edge(*edge)
            
            # Create new data with restored topology
            data_restored = self._rebuild_graph_data(data, G_restored, original_graph)
            
            # Evaluate
            new_error = self._compute_prediction_error(data_restored)
            error_reduction = current_error - new_error
            
            interventions.append({
                'type': 'edge_restoration',
                'components': [f'edge_{edge[0]}_{edge[1]}'],
                'details': [edge],
                'num_changes': 1,
                'current_error': float(current_error),
                'resulting_error': float(new_error),
                'error_reduction': float(error_reduction),
                'reduction_pct': float(error_reduction / current_error * 100) if current_error > 0 else 0,
                'success': new_error < error_threshold
            })
        
        # Try each failed node
        for node in failed_nodes:
            G_restored = failed_graph.copy()
            G_restored.add_node(node)
            
            # Restore edges connected to this node
            for u, v in original_graph.edges():
                if (u == node or v == node) and G_restored.has_node(u) and G_restored.has_node(v):
                    G_restored.add_edge(u, v)
            
            data_restored = self._rebuild_graph_data(data, G_restored, original_graph)
            new_error = self._compute_prediction_error(data_restored)
            error_reduction = current_error - new_error
            
            interventions.append({
                'type': 'node_restoration',
                'components': [f'node_{node}'],
                'details': [node],
                'num_changes': 1,
                'current_error': float(current_error),
                'resulting_error': float(new_error),
                'error_reduction': float(error_reduction),
                'reduction_pct': float(error_reduction / current_error * 100) if current_error > 0 else 0,
                'success': new_error < error_threshold
            })
        
        return interventions
    
    def _search_pair_restorations(self, data, failed_graph, original_graph,
                                  failed_edges, failed_nodes,
                                  error_threshold, current_error):
        """Try restoring pairs of components (limited to top candidates)."""
        interventions = []
        
        # Limit search space: only try top-K most important edges
        top_k = min(config.CFA_TOP_K_CANDIDATES, len(failed_edges))
        
        # Simple heuristic: edges with higher betweenness are more important
        edge_importance = {}
        try:
            edge_betweenness = nx.edge_betweenness_centrality(original_graph)
            for edge in failed_edges:
                edge_importance[edge] = edge_betweenness.get(edge, 0)
        except:
            # Fallback: random importance
            for edge in failed_edges:
                edge_importance[edge] = np.random.random()
        
        top_edges = sorted(failed_edges, key=lambda e: edge_importance.get(e, 0), reverse=True)[:top_k]
        
        # Try pairs
        max_pairs = 10  # Limit computation
        pairs_tried = 0
        
        for edge1, edge2 in combinations(top_edges, 2):
            if pairs_tried >= max_pairs:
                break
            
            G_restored = failed_graph.copy()
            G_restored.add_edge(*edge1)
            G_restored.add_edge(*edge2)
            
            data_restored = self._rebuild_graph_data(data, G_restored, original_graph)
            new_error = self._compute_prediction_error(data_restored)
            error_reduction = current_error - new_error
            
            interventions.append({
                'type': 'edge_pair_restoration',
                'components': [f'edge_{edge1[0]}_{edge1[1]}', f'edge_{edge2[0]}_{edge2[1]}'],
                'details': [edge1, edge2],
                'num_changes': 2,
                'current_error': float(current_error),
                'resulting_error': float(new_error),
                'error_reduction': float(error_reduction),
                'reduction_pct': float(error_reduction / current_error * 100) if current_error > 0 else 0,
                'success': new_error < error_threshold
            })
            
            pairs_tried += 1
        
        return interventions
    
    def _rebuild_graph_data(self, original_data, new_graph, reference_graph):
        """
        Create new PyG Data object with modified graph structure.
        
        Strategy: Keep node features, update edge_index based on new topology.
        """
        # Get edge list from new graph
        edges = list(new_graph.edges())
        
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Build new data object
        data_new = Data(
            x=original_data.x.clone(),
            edge_index=edge_index,
            y=original_data.y.clone()
        )
        
        # Handle batch dimension
        if hasattr(original_data, 'batch'):
            data_new.batch = original_data.batch.clone()
        else:
            data_new.batch = torch.zeros(data_new.x.size(0), dtype=torch.long)
        
        return data_new
    
    def _compute_prediction_error(self, data):
        """
        Compute prediction error.
        
        For classification: error = 1 - P(correct class)
        """
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            
            # output is log_softmax
            probs = torch.exp(output)
            
            # Get true label
            true_label = data.y.item() if data.y.dim() == 1 else data.y.squeeze().item()
            
            # Error = 1 - probability of correct class
            error = 1.0 - probs[0, true_label].item()
            
            return error
    
    def _format_results(self, interventions, current_error):
        """Format and rank intervention results."""
        # Separate successful and unsuccessful
        successful = [i for i in interventions if i['success']]
        unsuccessful = [i for i in interventions if not i['success']]
        
        # Sort successful by num_changes (minimal intervention first)
        successful.sort(key=lambda x: (x['num_changes'], -x['error_reduction']))
        
        # Sort unsuccessful by error_reduction (best attempt first)
        unsuccessful.sort(key=lambda x: -x['error_reduction'])
        
        # Combined ranking
        ranked = successful + unsuccessful
        
        return {
            'success': len(successful) > 0,
            'num_successful': len(successful),
            'num_total': len(interventions),
            'minimal_intervention': successful[0] if successful else (unsuccessful[0] if unsuccessful else None),
            'top_3_interventions': ranked[:3],
            'all_interventions': ranked,
            'current_error': float(current_error)
        }