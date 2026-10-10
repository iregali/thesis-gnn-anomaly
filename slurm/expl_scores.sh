#!/bin/bash
#SBATCH --job-name=expl_scores
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Per-seed test scores of the raw detectors at 1%, 2% and 5% contamination (static + deps + names + GuardDog),
# for the EXPLORATORY comparisons in config/exploratory_contamination.json (Methodology section 38):
# OCSVM, Isolation Forest, and OCSVM with per-group thresholds (needs PEER for the subgroups).
# Same settings selection and contamination draws as every baseline run, so the numbers must match section 21.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/baseline_protocol.py --levels 0.01 0.02 0.05 --variants naive --detectors OCSVM PEER IF \
    --feature-sets static+deps+names+GuardDog --boot 100 --tag _expl --save-scores
