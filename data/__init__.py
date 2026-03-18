"""
Data loading and preprocessing module.
"""

from .load_abilene import (
    build_abilene_graph,
    generate_synthetic_traffic,
    simulate_topology_failures,
    load_data
)

from .graph_dataset import (
    NetworkTrafficDataset,
    create_datasets
)

__all__ = [
    'build_abilene_graph',
    'generate_synthetic_traffic',
    'simulate_topology_failures',
    'load_data',
    'NetworkTrafficDataset',
    'create_datasets'
]