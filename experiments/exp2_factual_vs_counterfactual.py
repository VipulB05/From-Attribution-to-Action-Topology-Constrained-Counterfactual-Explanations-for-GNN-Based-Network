"""
Experiment 2: Factual vs Counterfactual RCA Comparison
Evaluate the effectiveness of CFA vs traditional factual explanations.
"""

import torch
import numpy as np
import pandas as pd
import json
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_abilene import build_abilene_graph, simulate_topology_failures
from explainability.integrated_rca import IntegratedRCA
import config


def run_factual_vs_counterfactual_experiment(model, test_dataset, device='cpu'):
    """
    Compare factual and counterfactual RCA approaches.
    
    Metrics:
    1. Intervention Success Rate: % of CF interventions that fix predictions
    2. Minimality: Avg number of components to restore
    3. Agreement Rate: % of cases where factual and CF agree
    4. Error Reduction: How much error decreases with intervention
    
    Returns:
        results: Dictionary with experimental results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: FACTUAL VS COUNTERFACTUAL RCA")
    print("="*80)
    
    # Build graphs
    original_graph = build_abilene_graph()
    
    results = {
        'factual_only': [],
        'counterfactual_only': [],
        'integrated': [],
        'case_studies': []
    }
    
    # Test different failure rates
    failure_rates = config.FAILURE_RATES
    
    for failure_rate in failure_rates:
        print(f"\n{'='*60}")
        print(f"Testing Failure Rate: {failure_rate*100:.0f}%")
        print(f"{'='*60}")
        
        # Simulate failures
        traffic = np.random.randn(len(test_dataset), len(original_graph.edges()))
        failed_graph, _, failed_edges = simulate_topology_failures(
            original_graph, traffic, failure_rate=failure_rate, seed=42
        )
        
        print(f"Failed components: {len(failed_edges)} edges")
        
        # Initialize RCA
        rca = IntegratedRCA(model, original_graph, device)
        
        # Evaluate on subset of test data
        n_samples = min(20, len(test_dataset))
        
        factual_scores = []
        cf_success = []
        cf_changes = []
        cf_error_reductions = []
        agreements = []
        case_studies = []
        
        for i in range(n_samples):
            data = test_dataset[i]
            
            # Get complete explanation
            explanation = rca.explain_complete(data, failed_graph, original_graph)
            
            # Factual metrics
            if explanation['factual']:
                factual_scores.append(explanation['factual'][0]['score'])
            
            # Counterfactual metrics
            cf_result = explanation['counterfactual']
            if cf_result['success']:
                minimal = cf_result['minimal_intervention']
                cf_success.append(1)
                cf_changes.append(minimal['num_changes'])
                cf_error_reductions.append(minimal['error_reduction'])
                
                # Save interesting case study
                if i < 3:  # Save first 3 cases
                    case_studies.append({
                        'failure_rate': failure_rate,
                        'sample_id': i,
                        'factual_cause': explanation['factual'][0] if explanation['factual'] else None,
                        'cf_intervention': minimal,
                        'agreement': explanation['integrated']['factual_cf_agreement'],
                        'recommendation': explanation['integrated']['recommendation']
                    })
            else:
                cf_success.append(0)
                cf_changes.append(0)
                cf_error_reductions.append(0)
            
            # Agreement
            agreements.append(1 if explanation['integrated']['factual_cf_agreement'] else 0)
        
        # Aggregate results for this failure rate
        results['factual_only'].append({
            'failure_rate': failure_rate,
            'avg_score': np.mean(factual_scores) if factual_scores else 0
        })
        
        results['counterfactual_only'].append({
            'failure_rate': failure_rate,
            'success_rate': np.mean(cf_success),
            'avg_changes': np.mean([c for c in cf_changes if c > 0]) if any(cf_changes) else 0,
            'avg_error_reduction': np.mean([e for e in cf_error_reductions if e > 0]) if any(cf_error_reductions) else 0
        })
        
        results['integrated'].append({
            'failure_rate': failure_rate,
            'agreement_rate': np.mean(agreements),
            'cf_success_rate': np.mean(cf_success)
        })
        
        results['case_studies'].extend(case_studies)
        
        print(f"\nResults for {failure_rate*100:.0f}% failure rate:")
        print(f"  CF Success Rate: {np.mean(cf_success):.2%}")
        avg_changes = np.mean([c for c in cf_changes if c > 0])
        print(f"  Avg Changes Needed: {avg_changes:.2f}" if any(cf_changes) else "  No successful interventions")
        print(f"  Agreement Rate: {np.mean(agreements):.2%}")
    
    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, 'factual_vs_cf_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_factual_vs_counterfactual_results(results, save_dir=config.FIGURES_DIR)
    
    # Generate LaTeX table
    generate_results_table(results)
    
    return results


def plot_factual_vs_counterfactual_results(results, save_dir='figures/'):
    """Generate comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    failure_rates = [r['failure_rate'] for r in results['counterfactual_only']]
    cf_success = [r['success_rate'] for r in results['counterfactual_only']]
    cf_changes = [r['avg_changes'] for r in results['counterfactual_only']]
    cf_error_red = [r['avg_error_reduction'] for r in results['counterfactual_only']]
    agreement = [r['agreement_rate'] for r in results['integrated']]
    
    # Plot 1: CF Success Rate
    axes[0, 0].plot(failure_rates, cf_success, marker='o', linewidth=2.5, markersize=10, color='steelblue')
    axes[0, 0].fill_between(failure_rates, cf_success, alpha=0.3, color='steelblue')
    axes[0, 0].set_xlabel('Failure Rate', fontsize=12)
    axes[0, 0].set_ylabel('Intervention Success Rate', fontsize=12)
    axes[0, 0].set_title('Counterfactual Intervention Success', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Plot 2: Minimality
    axes[0, 1].bar(range(len(failure_rates)), cf_changes, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Failure Rate', fontsize=12)
    axes[0, 1].set_ylabel('Avg Components to Restore', fontsize=12)
    axes[0, 1].set_title('Intervention Minimality', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(range(len(failure_rates)))
    axes[0, 1].set_xticklabels([f'{r*100:.0f}%' for r in failure_rates])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Error Reduction
    axes[1, 0].plot(failure_rates, cf_error_red, marker='s', linewidth=2.5, markersize=10, color='green')
    axes[1, 0].fill_between(failure_rates, cf_error_red, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Failure Rate', fontsize=12)
    axes[1, 0].set_ylabel('Avg Error Reduction', fontsize=12)
    axes[1, 0].set_title('Counterfactual Effectiveness', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Agreement Rate
    axes[1, 1].plot(failure_rates, agreement, marker='^', linewidth=2.5, markersize=10, color='purple')
    axes[1, 1].fill_between(failure_rates, agreement, alpha=0.3, color='purple')
    axes[1, 1].set_xlabel('Failure Rate', fontsize=12)
    axes[1, 1].set_ylabel('Agreement Rate', fontsize=12)
    axes[1, 1].set_title('Factual-Counterfactual Agreement', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'factual_vs_counterfactual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved: {save_dir}/factual_vs_counterfactual.png")


def generate_results_table(results):
    """Generate LaTeX table for paper."""
    print("\n" + "="*80)
    print("RESULTS TABLE (LaTeX format)")
    print("="*80)
    
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Factual vs Counterfactual RCA Performance}")
    print("\\label{tab:factual_vs_cf}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Failure Rate & CF Success & Avg Changes & Agreement \\\\")
    print("\\midrule")
    
    for i, rate in enumerate([r['failure_rate'] for r in results['counterfactual_only']]):
        cf_success = results['counterfactual_only'][i]['success_rate']
        cf_changes = results['counterfactual_only'][i]['avg_changes']
        agreement = results['integrated'][i]['agreement_rate']
        
        print(f"{rate*100:.0f}\\% & {cf_success:.2f} & {cf_changes:.2f} & {agreement:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*80)


if __name__ == "__main__":
    print("This script should be called from main.py")