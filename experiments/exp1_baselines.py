"""
Experiment 1: Baseline Model Comparison
Compare GNN models against traditional ML baselines.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_gnn import TrafficGCN, TrafficGAT, train_model_full, evaluate_gnn
from models.baselines import train_baseline_models
from utils.metrics import compute_all_metrics, print_results, compute_confusion_matrices
from utils.visualization import plot_baseline_comparison, plot_roc_curves, plot_confusion_matrices
import config


def run_baseline_experiment(train_dataset, val_dataset, test_dataset, device='cpu'):
    """
    Compare all baseline models and GNN models.
    
    Returns:
        all_results: Dictionary with results for each model
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE MODEL COMPARISON")
    print("="*80)
    
    all_results = {}
    
    # ============================================
    # Part 1: Train Traditional ML Baselines
    # ============================================
    print("\n[1/3] Training traditional ML baselines...")
    baseline_results = train_baseline_models(train_dataset, val_dataset, test_dataset)
    
    all_results['Logistic Regression'] = baseline_results['logistic']
    all_results['Random Forest'] = baseline_results['random_forest']
    all_results['Gradient Boosting'] = baseline_results['gbm']
    
    # ============================================
    # Part 2: Train GNN Models
    # ============================================
    print("\n[2/3] Training GNN models...")
    
    # GCN
    print("  Training GCN...")
    gcn_model = TrafficGCN(
        node_features=3,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        dropout=config.GNN_DROPOUT
    ).to(device)
    
    gcn_model = train_model_full(
        gcn_model, train_dataset, val_dataset,
        epochs=config.GNN_EPOCHS,
        lr=config.GNN_LEARNING_RATE,
        batch_size=config.GNN_BATCH_SIZE,
        device=device
    )
    
    # Evaluate GCN
    test_loader = DataLoader(test_dataset, batch_size=config.GNN_BATCH_SIZE, shuffle=False)
    gcn_results = evaluate_gnn(gcn_model, test_loader, device)
    all_results['GCN'] = gcn_results
    
    # Save GCN model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(gcn_model.state_dict(), os.path.join(config.MODELS_DIR, 'gcn_best.pt'))
    
    # GAT
    print("  Training GAT...")
    gat_model = TrafficGAT(
        node_features=3,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        heads=4,
        dropout=config.GNN_DROPOUT
    ).to(device)
    
    gat_model = train_model_full(
        gat_model, train_dataset, val_dataset,
        epochs=config.GNN_EPOCHS,
        lr=config.GNN_LEARNING_RATE,
        batch_size=config.GNN_BATCH_SIZE,
        device=device
    )
    
    # Evaluate GAT
    gat_results = evaluate_gnn(gat_model, test_loader, device)
    all_results['GAT'] = gat_results
    
    # Save GAT model
    torch.save(gat_model.state_dict(), os.path.join(config.MODELS_DIR, 'gat_best.pt'))
    
    # ============================================
    # Part 3: Compute and Display Metrics
    # ============================================
    print("\n[3/3] Computing metrics...")
    metrics_df = compute_all_metrics(all_results)
    print_results(metrics_df)
    
    # Save metrics
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(os.path.join(config.RESULTS_DIR, 'baseline_metrics.csv'), index=False)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    plot_baseline_comparison(metrics_df, 
                            save_path=os.path.join(config.FIGURES_DIR, 'baseline_comparison.png'))
    
    plot_roc_curves(all_results,
                   save_path=os.path.join(config.FIGURES_DIR, 'roc_curves.png'))
    
    confusion_matrices = compute_confusion_matrices(all_results)
    plot_confusion_matrices(all_results,
                           save_path=os.path.join(config.FIGURES_DIR, 'confusion_matrices.png'))
    
    print(f"\nAll results saved to {config.RESULTS_DIR}")
    print(f"All figures saved to {config.FIGURES_DIR}")
    
    return all_results, gcn_model, gat_model


if __name__ == "__main__":
    print("This script should be called from main.py")