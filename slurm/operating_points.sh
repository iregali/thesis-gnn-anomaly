#!/bin/bash
#SBATCH --job-name=operating_points
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# Revision (Oct 2026): detection across false-positive rates, from the saved scores (no training).
# Needs: baseline_protocol_scores_ops.csv (slurm/ops_scores.sh); contrastive_scores_triplet_gin_{tuned,n3}.csv,
# and their _c25 versions. Missing inputs are skipped.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/operating_points.py
