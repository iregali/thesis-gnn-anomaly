# Detecting Anomalous Components in Software Ecosystems via Self-Supervised Graph Neural Networks

Master's thesis project (Utrecht University, Artificial Intelligence).
Detection of **malicious packages** in the PyPI ecosystem with graph neural
networks trained **without labelled anomalies**, compared under one protocol
with rule-based static analysis (GuardDog) and classical feature-based
anomaly detectors.

> **Safety note.** The malicious data comes from archived real malware.
> Every extractor in this repository reads archives as text, in memory:
> nothing is ever installed, imported or executed. Do not unpack or install
> the malicious archives outside this pipeline.

## Overview

PyPI packages form a directed graph through their declared dependencies. Each
package release is a node; it is described by features extracted from its own
source distribution and connected to the packages it depends on. Detectors
learn what normal packages look like from unlabelled training data and score
previously unseen releases, which are inserted into the graph one at a time.

Methods compared under the same protocol:

- **Rule-based:** GuardDog source-code rules.
- **Feature-based detectors:** Isolation Forest, Local Outlier Factor,
  One-Class SVM (naive and robust variants, per-subgroup calibration), and a
  dependency-peer deviation detector.
- **Contrastive graph detector:** CoLA (target node vs. local subgraph).
- **One-class graph detectors:** Deep SVDD on GNN embeddings, with GIN, GAT,
  Transformer-style attention, TransformerConv and a GraphGPS-style encoder
  (local message passing + global attention over training packages).

## Data

Not included in the repository.

- **Malicious:** archives from malregistry; releases are grouped into
  campaign families, and validation/test are split **by family**, so test
  malware always comes from families never seen during model selection.
- **Normal:** a time-matched random sample of 15,000 PyPI projects
  (one release per project, source distributions only), matched to the
  yearly distribution of the malware.
- **Graph:** dependencies parsed from each archive's own declaration files
  with the same parser for both classes; dependency targets that are not
  sampled packages become *context nodes*. Isolated packages are kept.
- **Label-blind construction:** how a node enters the graph never depends on
  its label, and every feature is computed identically for both classes from
  the archive alone (no live registry metadata, download counts or release
  histories, which do not exist for removed malware).

Node features (51 columns): static archive features (package metadata, code
shape, capabilities, install-time behaviour, build setup), GuardDog rule
counts, numbers of declared dependencies, and name similarity to popular
packages (typosquatting).

## Evaluation protocol

- Validation and test packages are never part of the training graph; each is
  scored after insertion into the training graph, without links to other
  held-out packages.
- Training data: clean (main setting) and with 1 / 2 / 5 % unlabelled
  malware (contamination, sensitivity analysis), 3 seeds; identical draws for
  every method.
- Metrics: family-weighted ROC-AUC and detection at 0.5 / 1 / 2 % false
  positive rate, 95 % cluster-bootstrap intervals (malware resampled by
  family), precision at PyPI-derived base rates, and results per connectivity
  subgroup (connected vs. isolated packages), plain and per-subgroup
  calibrated.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Paths (`DATA_DIR`, `RESULTS_DIR`) are set at the top of the scripts and point
to the Snellius home directory; adjust them for another machine.

## Pipeline

Scripts are in `src/`, Slurm job scripts in `slurm/`.

| Step | Script(s) | Output |
|---|---|---|
| 1. Normal sample | `sample_normal.py` | time-matched sdists + manifest |
| 2. Static features | `extract_static_features.py` | `static_{malicious,normal}.csv` |
| 3. GuardDog features | `extract_guarddog.py` | `guarddog_{malicious,normal}.csv` |
| 4. Dependencies | `extract_dependencies.py` | `deps_*_edges.csv`, `deps_*_packages.csv` |
| 5. Name features | `fetch_popular_snapshots.py`, `make_typosquat_edges.py` | `name_features.csv`, `typosquat_pairs.csv` |
| 6. Campaigns and splits | `make_campaigns.py`, `make_splits.py`, `make_campaign_splits.py` | `campaigns.csv`, `splits_campaign_v2.csv` |
| 7. Graph bundles | `build_graph.py`, `check_bundles.py` | `graph_c{00,01,02,05}_s{seed}.pt` |
| 8. Baselines | `baseline_protocol.py` (uses `connectivity.py`) | `results/baseline_protocol_*.csv` |
| 9. Population checks | `fetch_release_history.py`, `connectivity.py`, `dep_combination_check.py` | release history, connectivity |
| 10. CoLA | `train_cola.py`, `diagnose_cola.py` | `results/cola_*.csv` |
| 11. One-class GNN | `train_ocgnn.py` (uses `gnn_eval.py`) | `results/ocgnn_*.csv` |

`losses.py` contains contrastive objectives, including the proposed
structure-graded loss. `baseline_experiment.py`, `build_features.py` and
`parse_malregistry.py` belong to an earlier version of the project (deps.dev
graph) and are kept for reference.

## Compute

Experiments run on the Snellius supercomputer (SURF). Feature extraction,
graph construction and baselines use CPU nodes (`rome`); GNN training uses
A100 MIG slices (`gpu_mig`). Downloads use the `staging` partition.

## Author

Irene Lara Galimi, Utrecht University, 2025–2026.
Supervisors: Prof. Slinger Jansen, Prof. Ioana Karnstedt-Hulpus.