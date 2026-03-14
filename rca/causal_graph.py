"""
Causal graph visualization: nodes = key features + target, edges = key drivers / causal structure.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_and_plot_causal_graph(
    feature_importance_dict,
    target_name="failure",
    top_n=8,
    save_path=None,
    title="Causal pathways to prediction failure",
):
    """
    Build a directed graph: top features -> target. Edge weight = importance.
    Plot and optionally save.
    """
    if isinstance(feature_importance_dict, dict):
        items = sorted(feature_importance_dict.items(), key=lambda x: -abs(x[1]))[:top_n]
    else:
        items = list(feature_importance_dict.items())[:top_n]

    G = nx.DiGraph()
    G.add_node(target_name)
    for feat, imp in items:
        G.add_node(feat)
        G.add_edge(feat, target_name, weight=float(imp))

    pos = nx.spring_layout(G, k=1.5, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1200)
    nx.draw_networkx_labels(G, pos, font_size=8)
    edges = G.edges()
    weights = [G[u][v].get("weight", 0.5) for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[2 + w * 3 for w in weights], alpha=0.7)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return G, plt.gcf()
