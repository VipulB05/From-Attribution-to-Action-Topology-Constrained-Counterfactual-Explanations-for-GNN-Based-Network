"""
Comprehensive metrics computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)


def compute_all_metrics(all_results):
    """
    Compute metrics for all models.
    
    Args:
        all_results: Dict of {model_name: {predictions, labels, probabilities, ...}}
    
    Returns:
        DataFrame with all metrics
    """
    metrics_list = []
    
    for model_name, results in all_results.items():
        has_precomputed_metrics = all(
            key in results for key in ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')
        )

        if has_precomputed_metrics:
            # Already computed (e.g., baselines)
            metrics = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1': results['f1'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'ROC-AUC': results['roc_auc']
            }
        else:
            # Compute from predictions/probabilities when not fully precomputed
            preds = np.array(results['predictions'])
            labels = np.array(results['labels'])
            probs = np.array(results['probabilities'])
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(labels, preds),
                'F1': f1_score(labels, preds, zero_division=0),
                'Precision': precision_score(labels, preds, zero_division=0),
                'Recall': recall_score(labels, preds, zero_division=0),
                'ROC-AUC': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
            }
        
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    return df


def print_results(metrics_df):
    """Pretty print results table."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("="*80)
    
    # Best model
    best_model = metrics_df.loc[metrics_df['F1'].idxmax(), 'Model']
    best_f1 = metrics_df['F1'].max()
    print(f"\nBest Model: {best_model} (F1 = {best_f1:.4f})")


def compute_confusion_matrices(all_results):
    """Compute confusion matrix for each model."""
    cms = {}
    
    for model_name, results in all_results.items():
        if 'predictions' in results:
            preds = np.array(results['predictions'])
            labels = np.array(results['labels'])
        else:
            preds = results['predictions']
            labels = results['labels']
        
        cm = confusion_matrix(labels, preds)
        cms[model_name] = cm
    
    return cms


def compute_rca_metrics(rca_results, ground_truth=None):
    """
    Compute RCA-specific metrics.
    
    Args:
        rca_results: List of RCA explanations
        ground_truth: Optional ground truth root causes
    
    Returns:
        Dictionary with RCA metrics
    """
    metrics = {
        'avg_top_score': 0,
        'avg_num_causes': 0,
        'score_std': 0
    }
    
    if not rca_results:
        return metrics
    
    # Compute average scores
    top_scores = [r[0]['score'] if r else 0 for r in rca_results]
    metrics['avg_top_score'] = np.mean(top_scores)
    metrics['score_std'] = np.std(top_scores)
    
    # Average number of causes identified
    num_causes = [len(r) for r in rca_results]
    metrics['avg_num_causes'] = np.mean(num_causes)
    
    # If ground truth available, compute accuracy
    if ground_truth is not None:
        correct = 0
        total = 0
        for i, r in enumerate(rca_results):
            if i < len(ground_truth) and r:
                if r[0]['id'] in ground_truth[i]:
                    correct += 1
                total += 1
        
        metrics['rca_accuracy'] = correct / total if total > 0 else 0
    
    return metrics


def compute_cf_metrics(cf_results):
    """
    Compute counterfactual-specific metrics.
    
    Args:
        cf_results: List of counterfactual results
    
    Returns:
        Dictionary with CF metrics
    """
    metrics = {
        'success_rate': 0,
        'avg_changes': 0,
        'avg_error_reduction': 0,
        'min_changes': float('inf'),
        'max_changes': 0
    }
    
    if not cf_results:
        return metrics
    
    # Success rate
    successes = [r['success'] if 'success' in r else False for r in cf_results]
    metrics['success_rate'] = np.mean(successes)
    
    # Changes needed
    successful_results = [r for r in cf_results if r.get('success', False)]
    if successful_results:
        changes = [r['minimal_intervention']['num_changes'] for r in successful_results]
        error_reductions = [r['minimal_intervention']['error_reduction'] for r in successful_results]
        
        metrics['avg_changes'] = np.mean(changes)
        metrics['avg_error_reduction'] = np.mean(error_reductions)
        metrics['min_changes'] = min(changes)
        metrics['max_changes'] = max(changes)
    
    return metrics


def print_detailed_report(model_name, results):
    """Print detailed classification report for a model."""
    print(f"\n{'='*80}")
    print(f"DETAILED REPORT: {model_name}")
    print(f"{'='*80}")
    
    preds = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Success', 'Failure']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(f"                 Predicted")
    print(f"                 Success  Failure")
    print(f"Actual Success   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"       Failure   {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    print("="*80)