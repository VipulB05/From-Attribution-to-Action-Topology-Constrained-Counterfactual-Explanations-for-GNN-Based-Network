# TCFA: Topology-Constrained Failure Attribution

Physically feasible counterfactual explanations for GNN-based traffic forecasting.

## What this is

Graph Neural Networks are increasingly used to predict failures in backbone
networks, but existing counterfactual explanation methods (CF-GNNExplainer,
GCFExplainer) recommend interventions with no regard for physical feasibility —
a suggested "fix" might disconnect the network, saturate a link's capacity, or
be unroutable. **TCFA** ranks candidate edge restorations under three hard/soft
physical constraints:

- **C1 — Connectivity**: the restored topology must remain connected
- **C2 — Capacity**: rerouted traffic must not exceed `1.5 × P95(historical load)`
- **C3 — Routing feasibility**: all origin–destination flows must remain
  routable under shortest-path routing

Candidates are ranked by `s(δ) = φ(δ) · max(0, P̂_base − P̂_δ)`, where φ
collapses to zero for any hard-constraint violation.

We also introduce **Fix@k** — the fraction of failures resolved by applying
the top-k ranked interventions — as an actionability metric distinct from
Hit@k/MRR, since a method can rank the correct edge first and still fail to
produce a physically usable fix.

## Results at a glance

Evaluated on **Abilene** (12 nodes, 15 edges) and **GÉANT** (23 nodes, 38
edges) across five held-out failure seeds:

| Metric | Abilene | GÉANT |
|---|---|---|
| Hit@1 / MRR | 100% / 1.000 | 100% / 1.000 |
| Fix@3 (TCFA) | 94% | 100% |
| Fix@3 (unconstrained ablation) | 0% | 0% |
| Constraint violation rate (unconstrained candidates) | 74.7% | 0.0% (sparse traffic regime) |

TCFA significantly outperforms a CF-GNNExplainer-inspired gradient-mask
baseline on actionability (McNemar p < 0.0001). Full methodology, ablations
(per-constraint, headroom sensitivity, restoration faithfulness/recovery
sufficiency), and statistical validation are in the paper.

## Repository contents

- `FINAL_v13.ipynb` — the complete pipeline: data loading, GRU traffic
  forecaster, GNN failure classifiers (GCN/GAT/GraphSAGE), TCFA constrained
  search, baselines, all ablations and significance tests. Run top to bottom.
- `data/` — Abilene and GÉANT topology + traffic matrices (SNDlib).
- `config.py` — hyperparameters and thresholds.
- `results/`, `results_geant/` — saved evaluation outputs (JSON).
- `figures/`, `figures_geant/` — generated plots.

## Running

Open `FINAL_v13.ipynb` and run cells in order. GPU recommended for GNN
training and the counterfactual search (constraint checking is the main
bottleneck on larger graphs like GÉANT).

## Paper

*Topology-Constrained Failure Attribution: Physically Feasible Counterfactual
Explanations for GNN-Based Traffic Forecasting*
Saanvi Manjunath, Sanjana Shenoy Katrisal, Vipul Rajesh Bohra, Bhaskarjyoti Das
Dept. of CSE (AIML), PES University, Bangalore

## Authors

Saanvi Manjunath · Sanjana Shenoy Katrisal · Vipul Rajesh Bohra · Bhaskarjyoti Das