# Detecting Anomalous Components in Software Ecosystems via Contrastive Self-Supervised Graph Neural Networks

Master's thesis project (Utrecht University, Artificial Intelligence).
Detection of **malicious releases** in the PyPI ecosystem with graph neural
networks trained **without labelled anomalies**, compared under one protocol
with rule-based static analysis (GuardDog) and classical feature-based
anomaly detectors.

> **Safety note.** The malicious data comes from archived real malware.
> Every extractor in this repository reads archives as text, in memory:
> nothing is ever installed, imported or executed. Do not unpack or install
> the malicious archives outside this pipeline.

## Overview

PyPI releases form a graph through their declared dependencies. Each release
is a node, described by 51 features extracted from its own source
distribution and connected to the packages it depends on. A GNN encoder is
trained without labels with a contrastive objective, frozen, and its
embeddings are scored by a classical anomaly detector. Held-out releases are
inserted into the training graph for scoring (inductive protocol).

Methods compared under the same protocol:

- **Rule-based:** GuardDog source-code rules.
- **Feature-based detectors:** Isolation Forest, Local Outlier Factor,
  One-Class SVM (plain, robust variant, per-subgroup calibration), and a
  supervised Random Forest as an upper reference.
- **Contrastive neighbourhood detector:** CoLA, reimplemented for the
  inductive protocol.
- **Contrastive framework (main method):** GIN, GAT, TransformerConv and
  GraphGPS encoders trained with InfoNCE, Triplet or DGI; embeddings scored
  by Isolation Forest, LOF or One-Class SVM; depth 0 (no neighbours) is the
  no-graph control.
- **Proposed structure-aware negatives** (`src/structure_rarity.py`), added to
  the three losses: *same-group negatives* (negatives only from the anchor's
  connectivity group, connected or isolated) and *rarity-shifted negatives*
  (a copy of the release, in its own neighbourhood, with one feature that is
  rare among training releases changed). Ablation conditions N1 to N4.
- **Control:** Deep SVDD on the same encoders (one-class objective).

## Data

Not included in the repository.

- **Malicious:** archives from malregistry; releases are grouped into
  campaign families, and validation / test / contamination pool are split
  **by family**, so test malware always comes from families never seen during
  model selection.
- **Normal:** a time-matched random sample of 15,000 PyPI projects
  (one release per project, source distributions only), matched to the
  yearly distribution of the malware.
- **Graph:** dependencies parsed from each archive's own declaration files
  with the same parser for both classes; dependency targets that are not
  sampled releases become *context nodes*. Isolated releases are kept.
- **Label-blind construction:** how a node enters the graph never depends on
  its label, and every feature is computed identically for both classes from
  the archive or the name alone (no live registry metadata, download counts or
  release histories, which do not exist for removed malware).

## Evaluation protocol

- Validation and test releases are never part of the training graph; each is
  scored after insertion into the training graph, without links to other
  held-out releases.
- Training data: clean (main setting) and with 1 / 2 / 5 % unlabelled
  malware (contamination, sensitivity analysis), 3 seeds; identical draws for
  every method.
- Settings chosen on validation only (partial ROC-AUC up to 5% FPR).
- Metrics: family-weighted ROC-AUC and share caught at 0.5 / 1 / 2 % false
  positive rate, precision at PyPI-derived base rates, results per
  connectivity subgroup, plain and per-subgroup calibrated.
- Statistics: 95% cluster-bootstrap intervals (malware resampled by family);
  paired two-level bootstrap with Holm correction for the comparisons fixed in
  advance in `config/primary_comparisons.json` (committed before the proposed
  loss was first run); everything else is reported as exploratory
  (`config/exploratory_contamination.json`).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Results were produced with Python 3.13.1 and the exact versions in
`requirements-lock.txt` (`pip install -r requirements-lock.txt` to reproduce
them). Paths (`DATA_DIR`, `RESULTS_DIR`) are set at the top of the scripts and
point to the Snellius home directory; adjust them for another machine.

## Pipeline

Scripts are in `src/`, Slurm job scripts in `slurm/`.

| Step | Script(s) | Slurm | Output |
|---|---|---|---|
| 1. Normal sample | `sample_normal.py` | `download_normal.sh` | time-matched sdists + manifest |
| 2. Static features | `extract_static_features.py` | `static_features.sh` | `static_{malicious,normal}.csv` |
| 3. GuardDog features | `extract_guarddog.py` | `gd_malicious.sh`, `gd_normal.sh` | `guarddog_{malicious,normal}.csv` |
| 4. Dependencies | `extract_dependencies.py` | | `deps_*_edges.csv`, `deps_*_packages.csv` |
| 5. Name features | `fetch_popular_snapshots.py`, `make_typosquat_edges.py` | | `name_features.csv`, `typosquat_pairs.csv` |
| 6. Campaigns and splits | `make_campaigns.py`, `make_splits.py`, `make_campaign_splits.py` | | `campaigns.csv`, `splits_campaign_v2.csv` |
| 7. Graph bundles | `build_graph.py`, `check_bundles.py` | `build_graph.sh`, `check_bundles.sh` | `graph_c{00,01,02,05}_s{seed}.pt` |
| 8. Reference detectors | `baseline_protocol.py` (uses `connectivity.py`) | `baseline_protocol.sh`, `baseline_conn.sh`, `baseline_robust.sh`, `baseline_resample.sh`, `baseline_split.sh` | `baseline_protocol_*` |
| 9. Population checks | `fetch_release_history.py`, `connectivity.py` | `release_history.sh` | release history, connectivity |
| 10. CoLA | `train_cola.py`, `diagnose_cola.py` | `train_cola.sh`, `cola_context.sh`, `diagnose_cola.sh`, `context_check.sh` | `cola_*` |
| 11. Base losses, default settings | `train_contrastive.py` | `train_contrastive.sh` | `contrastive_*_{loss}_{encoder}` |
| 12. Tuning of the base losses | `train_contrastive.py`, `select_tuning.py` | `tune_contrastive.sh` | `tuning_choice.csv` |
| 13. Base losses, tuned (reported) | `train_contrastive.py` | `train_tuned.sh` | `contrastive_*_{loss}_{encoder}_tuned` |
| 14. Proposed negatives: tuning | `structure_rarity.py`, `select_sr.py` | `tune_sr.sh` | `sr_choice.csv` |
| 15. Proposed negatives: N1 to N4 | `train_contrastive.py --strata --shift` | `train_sr.sh` | `contrastive_*_{loss}_{encoder}_n{1..4}` |
| 16. 2% and 5% contamination | as above, suffix `_c25` | `contamination_c25.sh` (run with `bash`) | `*_c25` |
| 17. Paired tests | `paired_compare.py`, `baseline_protocol.py --save-scores` | `primary_scores.sh`, `paired_compare.sh`, `expl_scores.sh`, `paired_compare_expl.sh` | `paired_comparisons*.csv` |
| 18. Controls | `train_contrastive.py`, `learning_curve_raw.py`, `train_ocgnn.py` | `capacity_control.sh`, `learning_curve.sh`, `learning_curve_raw.sh`, `ocgnn_rescale.sh`, `ocgnn_transformers.sh` | `*_d0_dim*`, `*_lc*`, `learning_curve_raw.csv`, `ocgnn_*` |
| 19. Diagnostics | `sensitivity_check.py`, `train_contrastive.py` | `sensitivity_check.sh`, `augmentation_checks.sh` | `sensitivity_check_*`, `*_mask*`, `*_tau*` |
| 20. Label noise | `label_noise_check.py` | `label_noise.sh` | `label_noise_summary.csv` |
| 21. Tables and figures | `make_tables_figures.py` | (no Slurm, runs anywhere) | `thesis_out/` |

Supporting modules: `gnn_eval.py` (shared evaluation), `train_ocgnn.py`
(encoders and neighbourhood sampling, imported by `train_contrastive.py`; its
own Deep SVDD training is the control), `structure_rarity.py` (proposed
negatives), `losses.py` (bilinear discriminator used by CoLA, plus an earlier
structure-graded loss draft that is not used in the thesis).

Analyses that led to the final design but are not part of the main results:
`complementarity_check.py`, `dep_combination_check.py`,
`artifact_twins_check.py`, `twins_vs_detector_check.py`, `relation_check.py`
(Slurm: `artifact_twins.sh`, `twins_vs_detector.sh`, `relation_check.sh`).
`slurm/train_ocgnn.sh` is the superseded first one-class run.
`baseline_experiment.py`, `build_features.py` and `parse_malregistry.py`
belong to an earlier version of the project (deps.dev graph) and are kept for
reference.

## Results and thesis tables

`results/` holds summary files only (metrics, settings, comparisons; one file
per result type, column `run` = original file name). Per-package score files
and the label-noise candidate list are never committed (`.gitignore`).
`python src/make_tables_figures.py` rebuilds every table and figure of the
results chapter from `results/` into `thesis_out/`, so no number in the
thesis is copied by hand.

## Compute

Experiments run on the Snellius supercomputer (SURF). Feature extraction,
graph construction and baselines use CPU nodes (`rome`); GNN training uses
GPU nodes (`gpu_mig` for the first runs, `gpu_h100` for the contrastive
runs). Downloads use the `staging` partition.

## Author

Irene Lara Galimi, Utrecht University, 2025–2026.
Supervisors: Prof. Slinger Jansen, Prof. Ioana Karnstedt-Hulpus.
