"""
Test loading real Abilene data.
"""

import numpy as np
from data.load_abilene import load_data
from data.graph_dataset import create_datasets
import config


def test_real_data_loading():
    """Test that real data loads correctly."""
    
    print("="*80)
    print("TEST: REAL ABILENE DATA LOADING")
    print("="*80)
    
    # Load data
    G, traffic, labels = load_data()
    
    # Verify shapes
    print(f"\nVerification:")
    print(f"  Graph nodes: {G.number_of_nodes()} (expected: 12)")
    print(f"  Graph edges: {G.number_of_edges()} (expected: 15)")
    print(f"  Traffic shape: {traffic.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Failure rate: {np.mean(labels)*100:.1f}%")
    
    # Check data validity
    assert G.number_of_nodes() == 12, "Wrong number of nodes"
    assert traffic.shape[0] == labels.shape[0], "Traffic and labels length mismatch"
    assert traffic.shape[1] == G.number_of_edges(), "Traffic edges != graph edges"
    assert np.all(traffic >= 0), "Negative traffic values found"
    assert np.all((labels == 0) | (labels == 1)), "Labels not binary"
    
    print(f"\n✓ All checks passed!")
    
    # Test dataset creation
    print(f"\nCreating PyG datasets...")
    train_ds, val_ds, test_ds = create_datasets(G, traffic, labels)
    
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    
    # Inspect first sample
    print(f"\nFirst sample:")
    sample = train_ds[0]
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Edge attr: {sample.edge_attr.shape}")
    print(f"  Label: {sample.y.item()}")
    
    print(f"\n" + "="*80)
    print("✓ REAL DATA LOADING TEST PASSED")
    print("="*80)


if __name__ == "__main__":
    test_real_data_loading()