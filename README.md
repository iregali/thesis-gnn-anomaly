# Detecting Anomalous Components in Software Ecosystems via Contrastive Self-Supervised Graph Neural Networks

Master's thesis project (Utrecht University, Artificial Intelligence).
A self-supervised contrastive GNN framework for detecting anomalous
(malicious or vulnerable) packages in the PyPI software ecosystem,
without relying on labelled anomalies during training.

## Overview

Packages in the PyPI ecosystem form a directed dependency graph.
This project trains a Graph Neural Network with a contrastive
self-supervised objective to learn structural representations of
package versions, then applies classical anomaly detectors to the
learned embeddings to rank packages by anomalousness. The central
contribution is a novel structure-aware contrastive loss tailored
to dependency graphs.

## Data

The experimental graph is built from the
[deps.dev](https://deps.dev) BigQuery public dataset (PyPI,
April 2026 snapshot). The subgraph used for experiments contains:

- 136,380 nodes (package versions)
- 496,525 directed dependency edges
- 39,033 confirmed-vulnerable nodes, 97,347 normal nodes

Each node has three feature groups:
- **Metadata**: in-degree, out-degree, version age, recency, is-latest
- **Code-level** (via [GuardDog](https://github.com/DataDog/guarddog)):
  code execution, network exfiltration, obfuscation, command abuse,
  sensitive access
- **Vulnerability** (via OSV): CVE count, maximum CVSS score

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

1. **Graph construction** — run the queries in `sql/` on BigQuery to
   build the subgraph, export to Cloud Storage.
2. **Feature engineering** — `build_features.py` computes metadata and
   vulnerability features; `extract_guarddog.py` extracts code-level
   features.
3. **Baseline** — `baseline_experiment.py` runs classical anomaly
   detectors on structural features.
4. **Training** — `train_contrastive.py` trains the contrastive GNN.

## Compute

Experiments run on the Snellius supercomputer (SURF). GNN training
uses GPU nodes; feature extraction uses CPU nodes. See `slurm/` for
job scripts.

## Author

Irene Lara Galimi — Utrecht University, 2025–2026.
Supervisors: Prof. Slinger Jansen, Prof. Ioana Karnstedt-Hulpus.