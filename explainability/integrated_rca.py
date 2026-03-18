"""
Integrated Root Cause Analysis
Combines factual + counterfactual explanations.
"""

import torch
import numpy as np
from .factual_rca import FactualRCA
from .contrastive_rca import ContrastiveRCA


class IntegratedRCA:
    """
    Complete RCA system: Factual (diagnosis) + Counterfactual (prescription)
    """
    
    def __init__(self, model, graph, device='cpu'):
        self.model = model
        self.graph = graph
        self.device = device
        
        # Initialize both explainers
        self.factual = FactualRCA(model, graph, device)
        self.contrastive = ContrastiveRCA(model, device)
    
    def explain_complete(self, data, failed_graph, original_graph, top_k=3):
        """
        Generate complete explanation with both factual and counterfactual.
        
        Returns:
            {
                'factual': {...},        # What caused the failure
                'counterfactual': {...}, # What's the minimal fix
                'integrated': {...}      # Combined insights
            }
        """
        print("\n" + "="*60)
        print("INTEGRATED ROOT CAUSE ANALYSIS")
        print("="*60)
        
        # Factual explanation
        print("\n[1/2] Generating factual explanation (diagnosis)...")
        factual_results = self.factual.explain(data, top_k=top_k)
        
        print(f"  Top root causes:")
        for i, cause in enumerate(factual_results, 1):
            print(f"    {i}. {cause['type'].upper()} {cause['id']}: score={cause['score']:.3f}")
        
        # Counterfactual explanation
        print("\n[2/2] Generating counterfactual explanation (prescription)...")
        counterfactual_results = self.contrastive.explain(
            data, failed_graph, original_graph
        )
        
        if counterfactual_results['success']:
            minimal = counterfactual_results['minimal_intervention']
            print(f"  ✓ Minimal fix found: {minimal['type']}")
            print(f"    Components: {minimal['components']}")
            print(f"    Error reduction: {minimal['error_reduction']:.3f} ({minimal['reduction_pct']:.1f}%)")
        else:
            print(f"  ✗ No single/pair intervention restores accuracy")
            if counterfactual_results['all_interventions']:
                best = counterfactual_results['all_interventions'][0]
                print(f"    Best attempt: {best['type']} reduces error by {best['reduction_pct']:.1f}%")
        
        # Integrate insights
        print("\n[3/3] Integrating factual + counterfactual...")
        integrated = self._integrate_explanations(factual_results, counterfactual_results)
        
        print(f"  Agreement: {'Yes' if integrated['factual_cf_agreement'] else 'No'}")
        print(f"  Recommendation: {integrated['recommendation']}")
        
        print("="*60 + "\n")
        
        return {
            'factual': factual_results,
            'counterfactual': counterfactual_results,
            'integrated': integrated
        }
    
    def _integrate_explanations(self, factual, counterfactual):
        """
        Combine factual and counterfactual insights.
        
        Key question: Do they agree on the root cause?
        """
        if not counterfactual['success']:
            return {
                'recommendation': 'No minimal intervention found - consider multiple repairs',
                'factual_cf_agreement': False,
                'confidence': 'low',
                'interpretation': 'System degradation too severe for single intervention'
            }
        
        minimal_fix = counterfactual['minimal_intervention']
        
        # Extract component IDs from both
        factual_ids = set()
        for cause in factual:
            if cause['type'] == 'node':
                factual_ids.add(f"node_{cause['id']}")
            elif cause['type'] == 'edge':
                factual_ids.add(f"edge_{cause['id'][0]}_{cause['id'][1]}")
        
        cf_ids = set(minimal_fix['components'])
        
        # Check agreement
        agreement = len(factual_ids & cf_ids) > 0
        
        # Generate recommendation
        if agreement:
            recommendation = (
                f"STRONG RECOMMENDATION: Restore {minimal_fix['components'][0]}. "
                f"Both factual and counterfactual analysis identify this component as critical. "
                f"Expected error reduction: {minimal_fix['error_reduction']:.2%}"
            )
            confidence = 'high'
            interpretation = (
                "Factual and counterfactual explanations converge on the same root cause. "
                "This indicates a clear, localizable failure."
            )
        else:
            recommendation = (
                f"MODERATE RECOMMENDATION: Restore {minimal_fix['components'][0]}. "
                f"Counterfactual analysis suggests this minimal intervention, "
                f"though factual analysis highlights {list(factual_ids)[0]}. "
                f"This may indicate indirect causation."
            )
            confidence = 'medium'
            interpretation = (
                f"Factual analysis identifies {list(factual_ids)[0]} as important, "
                f"but counterfactual suggests {minimal_fix['components'][0]} is the minimal fix. "
                "This suggests the visible failure symptom differs from the underlying structural cause."
            )
        
        return {
            'recommendation': recommendation,
            'factual_cf_agreement': agreement,
            'confidence': confidence,
            'interpretation': interpretation,
            'minimal_fix': minimal_fix,
            'top_factual_causes': factual[:3]
        }
    
    def batch_explain(self, data_list, failed_graphs, original_graph):
        """Explain multiple failures."""
        results = []
        for i, data in enumerate(data_list):
            failed_graph = failed_graphs[i] if isinstance(failed_graphs, list) else failed_graphs
            result = self.explain_complete(data, failed_graph, original_graph)
            results.append(result)
        return results