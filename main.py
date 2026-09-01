"""
Main execution script for Contrastive Root Cause Analysis.
Complete pipeline with baseline models, GNNs, and EdgeMLP comparison.
"""

import torch
import numpy as np
import os
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Import modules
from data.load_abilene import load_data
from data.graph_dataset import create_datasets
from models.traffic_gnn import TrafficGCN, TrafficGAT, train_model_full, evaluate_gnn
from models.baselines import train_baseline_models
from utils.metrics import compute_all_metrics, print_results
from utils.visualization import plot_all_figures
from explainability.factual_rca import FactualRCA

import config


def train_edge_mlp(train_dataset, val_dataset, test_dataset, device='cpu'):
    """
    Train simple MLP on edge traffic (no graph structure).
    This tests if the problem is with node feature aggregation.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    print("\n[DIAGNOSTIC TEST] Training EdgeMLP (direct edge traffic, no GNN)...")
    
    # Extract edge traffic and labels
    train_X = torch.stack([data.edge_attr[:15, 0] for data in train_dataset])
    train_y = torch.tensor([data.y.item() for data in train_dataset])
    
    val_X = torch.stack([data.edge_attr[:15, 0] for data in val_dataset])
    val_y = torch.tensor([data.y.item() for data in val_dataset])
    
    test_X = torch.stack([data.edge_attr[:15, 0] for data in test_dataset])
    test_y = torch.tensor([data.y.item() for data in test_dataset])
    
    # Define simple MLP
    class EdgeMLP(nn.Module):
        def __init__(self, n_edges=15, hidden_dim=128):
            super().__init__()
            self.fc1 = nn.Linear(n_edges, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, 2)
            self.dropout = nn.Dropout(0.4)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)
    
    model = EdgeMLP(n_edges=15, hidden_dim=128).to(device)
    
    # Compute class weights
    class_counts = torch.bincount(train_y)
    weights = len(train_y) / (2 * class_counts.float())
    print(f"  Class weights: {weights.tolist()}")
    
    # Training setup
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_f1 = 0
    patience = 0
    
    for epoch in range(150):
        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = F.nll_loss(out, y, weight=weights.to(device))
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                out = model(X)
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_labels.extend(y.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            best_state = model.state_dict().copy()
        else:
            patience += 1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if patience >= 30:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            out = model(X)
            probs = torch.exp(out)
            test_preds.extend(out.argmax(dim=1).cpu().numpy())
            test_labels.extend(y.numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
    
    print(f"  Best validation F1: {best_f1:.4f}")
    
    return {
        'predictions': test_preds,
        'labels': test_labels,
        'probabilities': test_probs,
        'accuracy': accuracy_score(test_labels, test_preds),
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0)
    }


def main():
    """Run complete experimental pipeline."""
    
    print("="*80)
    print("CONTRASTIVE ROOT CAUSE ANALYSIS FOR NETWORK TRAFFIC PREDICTION")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    print("\n[1/6] Loading data...")
    
    G, traffic, labels = load_data()
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        G, traffic, labels, 
        train_ratio=config.TRAIN_RATIO, 
        val_ratio=config.VAL_RATIO,
        random_state=config.RANDOM_STATE
    )
    
    # ============================================
    # STEP 2: Train Baseline Models
    # ============================================
    print("\n[2/6] Training baseline models...")
    baseline_results = train_baseline_models(train_dataset, val_dataset, test_dataset)
    
    # ============================================
    # STEP 3: Train GNN Models
    # ============================================
    print("\n[3/6] Training GNN models...")
    
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
    
    # Save GAT model
    torch.save(gat_model.state_dict(), os.path.join(config.MODELS_DIR, 'gat_best.pt'))
    
    # ============================================
    # STEP 4: EdgeMLP Diagnostic Test
    # ============================================
    print("\n[4/6] Running diagnostic test...")
    edge_mlp_results = train_edge_mlp(train_dataset, val_dataset, test_dataset, device)
    
    # ============================================
    # STEP 5: Compute Metrics
    # ============================================
    print("\n[5/6] Computing metrics...")
    
    all_results = {
        'Logistic Regression': baseline_results['logistic'],
        'Random Forest': baseline_results['random_forest'],
        'Gradient Boosting': baseline_results['gbm'],
        'GCN': gcn_results,
        'GAT': gat_results,
        'EdgeMLP': edge_mlp_results
    }
    
    metrics = compute_all_metrics(all_results)
    print_results(metrics)
    
    # Print diagnostic analysis
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS")
    print("="*80)
    print(f"EdgeMLP F1 Score: {edge_mlp_results['f1']:.4f}")
    print(f"GCN F1 Score: {gcn_results['f1']:.4f}")
    print(f"GAT F1 Score: {gat_results['f1']:.4f}")
    print()
    
    if edge_mlp_results['f1'] > 0.5 and gcn_results['f1'] < 0.3:
        print("⚠️  DIAGNOSIS: Node feature aggregation is the problem!")
        print("   EdgeMLP (no graph) works well, but GNNs fail.")
        print("   → Issue: Node features don't capture enough temporal variation")
        print("   → Solution: Add temporal features or use edge-level GNN")
    elif edge_mlp_results['f1'] > 0.5:
        print("✓  EdgeMLP and GNNs both work - features are adequate")
    else:
        print("⚠️  Both EdgeMLP and GNNs struggle - data may have inherent limitations")
        print("   → Consider: More data, different features, or different problem formulation")
    print("="*80)
    
    # Save metrics
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics.to_csv(os.path.join(config.RESULTS_DIR, 'baseline_metrics.csv'), index=False)
    
    # ============================================
    # STEP 6: Generate Figures
    # ============================================
    print("\n[6/6] Generating figures...")
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Initialize factual RCA for visualization
    factual_rca = FactualRCA(gcn_model, G, device)
    
    plot_all_figures(
        all_results, 
        G, 
        factual_rca,
        test_dataset,
        save_dir=config.FIGURES_DIR
    )
    
    print(f"\n✅ All results saved!")
    print(f"   - Metrics: {config.RESULTS_DIR}")
    print(f"   - Figures: {config.FIGURES_DIR}")
    print(f"   - Models: {config.MODELS_DIR}")
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    
    # Final recommendation
    print("\n📊 NEXT STEPS:")
    if edge_mlp_results['f1'] > gcn_results['f1'] + 0.2:
        print("1. Enhance node features with temporal information")
        print("2. Or switch to edge-level GNN architecture")
        print("3. Current GNN architecture needs improvement")
    else:
        print("1. Proceed with current architecture")
        print("2. Implement CFA (Contrastive Failure Attribution)")
        print("3. Run full experiments and write paper")


if __name__ == "__main__":
    main()
