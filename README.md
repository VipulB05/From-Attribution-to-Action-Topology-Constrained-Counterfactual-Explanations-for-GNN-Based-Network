# Explainable Root Cause Analysis for GNN Traffic Prediction Failures

Pipeline for **explainability** and **root cause analysis** of (GNN) traffic prediction failures, aligned with the course project one-pager (Abilene/GÉANT-style networks, topology-aware attribution).

## Features

- **Data**: Abilene-like synthetic traffic (12 nodes, topology + link loads) or CSV from SNDlib-style exports
- **Preprocessing**: Missing value imputation, standardization, train/test split
- **Baseline**: Logistic regression, Random Forest, or Gradient Boosting for failure classification
- **Explainability**:
  - Feature importance (model-specific + permutation)
  - SHAP (local/global)
  - LIME (local)
  - Counterfactual examples
- **Root cause analysis**:
  - Key drivers of the target (failure)
  - DoWhy-based causal estimation
  - Sensitivity analysis
  - Causal graph visualization
- **Evaluation**: Accuracy, F1, ROC-AUC; interpretability (consistency, stability)

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Script (all steps):**
```bash
python run_pipeline.py
```

**Notebook (step-by-step):**
Open `explainable_rca_pipeline.ipynb` and run all cells (run from project root).

## Outputs

Generated in `outputs/`:

- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`
- `shap_summary.png`
- `sensitivity_analysis.png`
- `causal_graph.png`

## Project layout

```
├── config.py              # Paths, Abilene topology, thresholds
├── run_pipeline.py        # End-to-end script
├── explainable_rca_pipeline.ipynb
├── data/
│   └── load_and_preprocess.py
├── models/
│   └── baseline.py
├── explainability/
│   ├── feature_importance.py
│   ├── shap_explanations.py
│   ├── lime_explanations.py
│   └── counterfactuals.py
├── rca/
│   ├── root_cause.py
│   └── causal_graph.py
├── evaluation/
│   └── metrics.py
└── visualization/
    └── plots.py
```

## Data

- **Synthetic**: Abilene-like 12-node graph and link-level traffic; “failure” = relative prediction error above a threshold.
- **Custom**: Place a CSV with `link_*` (and optional `time_idx`) in `data/` and pass its path to `load_abilene_like_data(data_path="data/your_file.csv")`. If the CSV has no `failure`/`pred_error` columns, they are simulated.

## References (from one-pager)

- GNNExplainer; TraffExplainer; Graph WaveNet; DCRNN.
