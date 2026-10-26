#!/bin/bash
#SBATCH --job-name=paired_ops
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Revision (Oct 2026): EXPLORATORY paired comparisons R1-R9 at 0.5% and 2% FPR
# (config/exploratory_operating_points.json). Needs baseline_protocol_scores_ops.csv (slurm/ops_scores.sh)
# and the N3 runs _triplet_gin_n3 (1%) and _triplet_gin_n3_c25 (2%, 5%).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/paired_compare.py --spec config/exploratory_operating_points.json --tag _ops
