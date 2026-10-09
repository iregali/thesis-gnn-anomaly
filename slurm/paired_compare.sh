#!/bin/bash
#SBATCH --job-name=paired_compare
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Pre-registered paired comparisons with Holm correction (Methodology section 38).
# Needs: baseline_protocol_scores_primary.csv (slurm/primary_scores.sh), the tuned runs (_*_gin_tuned)
# and the new-loss runs (_*_gin_n1 ... _n4). Every tag in the spec must exist, or the job stops.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/paired_compare.py --spec config/primary_comparisons.json
