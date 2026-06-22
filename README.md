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