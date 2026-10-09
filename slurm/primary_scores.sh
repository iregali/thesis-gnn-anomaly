#!/bin/bash
#SBATCH --job-name=primary_scores
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Per-seed test scores of the raw OCSVM (static + deps + names + GuardDog, naive, clean training),
# the reference in comparisons P7 / P8 of config/primary_comparisons.json (Methodology section 38).
# Same settings selection as every baseline run, so its numbers must match section 21 (0.963 / 73.9%).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/baseline_protocol.py --levels 0 --variants naive --detectors OCSVM \
    --feature-sets static+deps+names+GuardDog --boot 100 --tag _primary --save-scores
