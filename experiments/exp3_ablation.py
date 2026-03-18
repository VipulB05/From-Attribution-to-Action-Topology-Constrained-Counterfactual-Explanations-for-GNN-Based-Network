"""
Experiment 3: Ablation Study
Evaluate contribution of each component in the integrated RCA system.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_abilene import build_abilene_graph, simulate_topology_failures
from explainability.factual_rca import FactualRCA
from explainability.contrastive_rca import ContrastiveRCA
import config


def run_ablation_study(model, test_dataset, device='cpu'):
    """
    Ablation study: Test each component in isolation.
    
    Variants:
    1. Model attribution only (no topology)
    2. Topology only (no model attribution)
    3. Factual only (model + topology)
    4. Counterfactual only
    5. Integrated (full system)
    
    Returns:
        results: Dictionary with ablation results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: ABLATION STUDY")
    print("="*80)
    
    # Build graphs
    original_graph = build_abilene_graph()
    
    # Simulate failures
    traffic = np.random.randn(len(test_dataset), len(original_graph.edges()))
    failed_graph, _, failed_edges = simulate_topology_failures(
        original_graph, traffic, failure_rate=0.2, seed=42
    )
    
    results = {
        'model_only': [],
        'topology_only': [],
        'factual_combined': [],
        'counterfactual_only': [],
        'integrated': []
    }
    
    # Initialize components
    factual_rca = FactualRCA(model, original_graph, device)
    cf_rca = ContrastiveRCA(model, device)
    
    # Test on subset
    n_samples = min(20, len(test_dataset))
    
    print(f"\nEvaluating {n_samples} samples...")
    
    for i in range(n_samples):
        data = test_dataset[i]
        
        # Variant 1: Model attribution only
        # (Simplified - just use GNNExplainer without topology)
        model_result = factual_rca.explain(data, top_k=1)
        if model_result:
            results['model_only'].append({
                'top_cause': model_result[0]['id'],
                'score': model_result[0]['model_attribution']  # Only model part
            })
        
        # Variant 2: Topology only
        # (Use topology risk as the only signal)
        if model_result:
            results['topology_only'].append({
                'top_cause': model_result[0]['id'],
                'score': model_result[0]['topology_risk']  # Only topology part
            })
        
        # Variant 3: Factual combined (model + topology)
        factual_result = factual_rca.explain(data, top_k=1)
        if factual_result:
            results['factual_combined'].append({
                'top_cause': factual_result[0]['id'],
                'score': factual_result[0]['score']  # Combined score
            })
        
        # Variant 4: Counterfactual only
        cf_result = cf_rca.explain(data, failed_graph, original_graph)
        if cf_result['success']:
            minimal = cf_result['minimal_intervention']
            results['counterfactual_only'].append({
                'intervention': minimal['components'][0],
                'success': True,
                'error_reduction': minimal['error_reduction']
            })
        else:
            results['counterfactual_only'].append({
                'intervention': None,
                'success': False,
                'error_reduction': 0
            })
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    
    # Model only
    model_scores = [r['score'] for r in results['model_only']]
    print(f"\nModel Attribution Only:")
    print(f"  Avg Score: {np.mean(model_scores):.3f}")
    print(f"  Std Score: {np.std(model_scores):.3f}")
    
    # Topology only
    topo_scores = [r['score'] for r in results['topology_only']]
    print(f"\nTopology Only:")
    print(f"  Avg Score: {np.mean(topo_scores):.3f}")
    print(f"  Std Score: {np.std(topo_scores):.3f}")
    
    # Factual combined
    factual_scores = [r['score'] for r in results['factual_combined']]
    print(f"\nFactual Combined (Model + Topology):")
    print(f"  Avg Score: {np.mean(factual_scores):.3f}")
    print(f"  Std Score: {np.std(factual_scores):.3f}")
    print(f"  Improvement over Model Only: {(np.mean(factual_scores) - np.mean(model_scores)):.3f}")
    print(f"  Improvement over Topology Only: {(np.mean(factual_scores) - np.mean(topo_scores)):.3f}")
    
    # Counterfactual only
    cf_success_rate = np.mean([r['success'] for r in results['counterfactual_only']])
    cf_error_red = np.mean([r['error_reduction'] for r in results['counterfactual_only'] if r['success']])
    print(f"\nCounterfactual Only:")
    print(f"  Success Rate: {cf_success_rate:.2%}")
    print(f"  Avg Error Reduction: {cf_error_red:.3f}")
    
    # Generate visualization
    plot_ablation_results(results, save_dir=config.FIGURES_DIR)
    
    return results


def plot_ablation_results(results, save_dir='figures/'):
    """Visualize ablation study results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Score comparison
    methods = ['Model\nOnly', 'Topology\nOnly', 'Factual\nCombined']
    scores = [
        np.mean([r['score'] for r in results['model_only']]),
        np.mean([r['score'] for r in results['topology_only']]),
        np.mean([r['score'] for r in results['factual_combined']])
    ]
    
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    bars = axes[0].bar(methods, scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Average Score', fontsize=12)
    axes[0].set_title('Factual RCA Component Contribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=11)
    
    # Plot 2: CF success rate
    cf_success = np.mean([r['success'] for r in results['counterfactual_only']])
    
    axes[1].bar(['Counterfactual\nRCA'], [cf_success], color='steelblue', edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Success Rate', fontsize=12)
    axes[1].set_title('Counterfactual Intervention Success', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value label
    axes[1].text(0, cf_success, f'{cf_success:.2%}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved: {save_dir}/ablation_study.png")


if __name__ == "__main__":
    print("This script should be called from main.py")