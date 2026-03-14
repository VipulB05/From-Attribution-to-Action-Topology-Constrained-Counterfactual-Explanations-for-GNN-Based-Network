"""
Full pipeline: Explainable Root Cause Analysis for (GNN) Traffic Prediction Failures.
Load data -> Preprocess -> Train baseline -> Explainability (SHAP, LIME, counterfactuals)
-> Root cause analysis (DoWhy, sensitivity, causal graph) -> Evaluate -> Visualize.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config
from data.load_and_preprocess import load_abilene_like_data, preprocess_dataset, get_train_test_split
from models.baseline import train_baseline_model, get_baseline_predictions
from explainability.feature_importance import get_model_importance, get_permutation_importance
from explainability.shap_explanations import compute_shap_values, shap_summary_plot
from explainability.lime_explanations import get_lime_explanations
from explainability.counterfactuals import generate_counterfactuals
from rca.root_cause import identify_key_drivers, run_causal_analysis, sensitivity_analysis
from rca.causal_graph import build_and_plot_causal_graph
from evaluation.metrics import compute_standard_metrics, compute_interpretability_metrics
from visualization.plots import plot_confusion_matrix, plot_roc_curve, plot_feature_importance


def main():
    print("=" * 60)
    print("Explainable Root Cause Analysis Pipeline")
    print("(Traffic Prediction Failure - Abilene-like)")
    print("=" * 60)

    # --- 1. Load and preprocess ---
    print("\n[1] Loading and preprocessing data...")
    df, graph, feature_cols, target_name = load_abilene_like_data(
        data_path=None, n_timesteps=config.N_TIMESTAMPS
    )
    print(f"    Loaded {len(df)} samples, {len(feature_cols)} features, target={target_name}")

    X, y, feature_names, scaler = preprocess_dataset(
        df, feature_cols, target_name, impute=True, normalize=True
    )
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, stratify=y)
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # --- 2. Train baseline ---
    print("\n[2] Training baseline model...")
    model, task = train_baseline_model(
        X_train, y_train, task="classification", model_type=config.BASELINE_MODEL
    )
    y_pred, y_proba = get_baseline_predictions(model, X_test, task)
    metrics = compute_standard_metrics(y_test, y_pred, y_proba, task=task)
    print("    Test metrics:", metrics)

    # --- 3. Explainability ---
    print("\n[3] Explainability...")
    model_imp = get_model_importance(model, feature_names)
    perm_imp_df = get_permutation_importance(
        model, X_test, y_test, n_repeats=5, scoring="accuracy"
    )
    print("    Top 5 model importance:", list(get_model_importance(model, feature_names).items())[:5])

    # SHAP (on a subset for speed)
    n_shap = min(config.SHAP_SAMPLES, len(X_test))
    X_explain = X_test.sample(n_shap, random_state=config.RANDOM_STATE)
    shap_vals, _ = compute_shap_values(model, X_train, X_explain, task="classification", max_background=50)
    shap_fig = None
    if shap_vals is not None:
        shap_summary_plot(
            shap_vals, X_explain, feature_names=feature_names,
            save_path=os.path.join(config.OUTPUT_DIR, "shap_summary.png")
        )
        print("    SHAP summary saved to outputs/shap_summary.png")

    # LIME on one instance
    lime_exp, lime_weights = get_lime_explanations(
        model, X_test, 0, feature_names=feature_names, task="classification"
    )
    if lime_weights:
        print("    LIME top features (instance 0):", lime_weights[:5])

    # Counterfactuals for one failure instance
    fail_idx = np.where(y_test == 1)[0]
    if len(fail_idx) > 0:
        idx = fail_idx[0]
        cf_df, cf_results = generate_counterfactuals(
            model, X_test.iloc[idx], feature_names, target_class=0, n_cf=config.N_COUNTERFACTUALS
        )
        print("    Counterfactuals generated for one failure instance.")

    # --- 4. Root cause analysis ---
    print("\n[4] Root cause analysis...")
    key_drivers = identify_key_drivers(model_imp, top_k=10)
    print("    Key drivers:", [k for k, _ in key_drivers[:5]])

    sensitivity = sensitivity_analysis(
        model, X_test, y_test, feature_names, n_perturbations=15, noise_scale=0.1
    )
    top_sens = sorted(sensitivity.items(), key=lambda x: -x[1])[:5]
    print("    Top sensitivity:", top_sens)

    # Causal (DoWhy) on a subset of features
    df_causal = pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name=target_name)], axis=1)
    treatment = [feature_names[0]] if feature_names else []
    if len(feature_names) > 3:
        common_causes = feature_names[1:4]
    else:
        common_causes = feature_names[1:] if len(feature_names) > 1 else []
    if treatment and common_causes:
        est, _ = run_causal_analysis(df_causal, treatment, target_name, common_causes=common_causes)
        if est is not None:
            print("    Causal estimate (DoWhy):", repr(est)[:200])

    # Causal graph
    G_causal, fig_causal = build_and_plot_causal_graph(
        model_imp, target_name=target_name, top_n=8,
        save_path=os.path.join(config.OUTPUT_DIR, "causal_graph.png"),
        title="Causal pathways to prediction failure",
    )
    print("    Causal graph saved to outputs/causal_graph.png")

    # --- 5. Evaluation & interpretability metrics ---
    print("\n[5] Evaluation...")
    interp_metrics = compute_interpretability_metrics(
        model_imp, perm_imp_df, shap_vals, X_explain, feature_names
    )
    print("    Interpretability metrics:", interp_metrics)

    # --- 6. Visualizations ---
    print("\n[6] Saving visualizations...")
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"),
    )
    if y_proba is not None:
        proba_pos = y_proba[:, 1] if hasattr(y_proba, "ndim") and y_proba.ndim > 1 else np.array(y_proba).ravel()
        plot_roc_curve(y_test, proba_pos, save_path=os.path.join(config.OUTPUT_DIR, "roc_curve.png"))
    plot_feature_importance(
        model_imp, top_k=15,
        save_path=os.path.join(config.OUTPUT_DIR, "feature_importance.png"),
        title="Model feature importance (root cause drivers)",
    )
    plt_sens = {k: v for k, v in sorted(sensitivity.items(), key=lambda x: -x[1])[:15]}
    plot_feature_importance(
        plt_sens, top_k=15,
        save_path=os.path.join(config.OUTPUT_DIR, "sensitivity_analysis.png"),
        title="Sensitivity analysis",
    )

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs in:", config.OUTPUT_DIR)
    print("=" * 60)
    return {
        "metrics": metrics,
        "interpretability": interp_metrics,
        "key_drivers": key_drivers,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()
