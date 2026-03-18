"""
Visualization functions for paper figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11


def plot_baseline_comparison(metrics_df, save_path=None):
    """Figure 1: Bar chart comparing all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    axes[0].bar(metrics_df['Model'], metrics_df['Accuracy'], color='steelblue', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(metrics_df['Model'], metrics_df['Accuracy'])):
        axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)
    
    # F1 Score comparison
    axes[1].bar(metrics_df['Model'], metrics_df['F1'], color='coral', edgecolor='black')
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Add value labels
    for i, (model, f1) in enumerate(zip(metrics_df['Model'], metrics_df['F1'])):
        axes[1].text(i, f1 + 0.02, f'{f1:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()
    return fig


def plot_roc_curves(all_results, save_path=None):
    """Figure 2: ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        labels = np.array(results['labels'])
        probs = np.array(results['probabilities'])
        
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2.5, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrices(all_results, save_path=None):
    """Figure 3: Confusion matrices for all models."""
    n_models = len(all_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        labels = np.array(results['labels'])
        preds = np.array(results['predictions'])
        
        cm = confusion_matrix(labels, preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[idx], cbar=False, square=True,
                   xticklabels=['Success', 'Failure'],
                   yticklabels=['Success', 'Failure'])
        axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide extra subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()
    return fig


def plot_network_attribution(G, rca, sample_data, save_path=None):
    """Figure 4: Network visualization with RCA attribution."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get root causes
    root_causes = rca.explain(sample_data, top_k=5)
    
    # Node colors based on importance
    node_colors = []
    for node in G.nodes():
        importance = 0.0
        for rc in root_causes:
            if rc['type'] == 'node' and rc['id'] == node:
                importance = rc['score']
                break
        node_colors.append(importance)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, ax=ax)
    
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors,
        node_size=800,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        ax=ax
    )
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Colorbar
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Root Cause Importance', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_title('Network Topology with Root Cause Attribution', 
                fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    textstr = "Top Root Causes:\n"
    for i, rc in enumerate(root_causes[:3], 1):
        textstr += f"{i}. Node {rc['id']} (score: {rc['score']:.3f})\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()
    return fig


def plot_all_figures(all_results, G, rca, test_dataset, save_dir='figures/'):
    """Generate all figures for the paper."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute metrics
    from utils.metrics import compute_all_metrics
    metrics_df = compute_all_metrics(all_results)
    
    print("  Generating Figure 1: Baseline Comparison...")
    plot_baseline_comparison(metrics_df, save_path=os.path.join(save_dir, 'baseline_comparison.png'))
    
    print("  Generating Figure 2: ROC Curves...")
    plot_roc_curves(all_results, save_path=os.path.join(save_dir, 'roc_curves.png'))
    
    print("  Generating Figure 3: Confusion Matrices...")
    plot_confusion_matrices(all_results, save_path=os.path.join(save_dir, 'confusion_matrices.png'))
    
    print("  Generating Figure 4: Network Attribution...")
    sample_data = test_dataset[0]
    plot_network_attribution(G, rca, sample_data, save_path=os.path.join(save_dir, 'network_attribution.png'))
    
    print(f"\nAll figures saved to {save_dir}")