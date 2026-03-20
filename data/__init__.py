"""
Data loading and preprocessing module.
"""

from .load_abilene import (
    build_abilene_graph,
    load_abilene_traffic_matrices,
    od_matrix_to_edge_traffic,
    generate_failure_labels,
    simulate_topology_failures,
    load_data
)

from .graph_dataset import (
    NetworkTrafficDataset,
    create_datasets
)

__all__ = [
    'build_abilene_graph',
    'load_abilene_traffic_matrices',
    'od_matrix_to_edge_traffic',
    'generate_failure_labels',
    'simulate_topology_failures',
    'load_data',
    'NetworkTrafficDataset',
    'create_datasets'
]