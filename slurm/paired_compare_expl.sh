#!/bin/bash
#SBATCH --job-name=paired_expl
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# EXPLORATORY paired comparisons at 1%, 2% and 5% contamination (config/exploratory_contamination.json).
# Needs: baseline_protocol_scores_expl.csv (slurm/expl_scores.sh), the tuned and N3 runs at 1%
# (_triplet_{gin,gat,tconv,gps}_{tuned,n3}) and the 2%/5% runs (_triplet_gin_{tuned,n3}_c25).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/paired_compare.py --spec config/exploratory_contamination.json --tag _expl
