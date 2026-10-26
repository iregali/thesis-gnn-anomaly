#!/bin/bash
#SBATCH --job-name=ops_scores
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Revision (Oct 2026): per-seed test scores (validation scores too once point 4 is applied) of the raw detectors at every contamination level,
# for src/operating_points.py (detection across false-positive rates) and config/exploratory_operating_points.json.
# OCSVM, Isolation Forest, and OCSVM with per-group thresholds (needs PEER for the subgroups).
# Same settings selection and contamination draws as every baseline run: the 1% FPR numbers must match
# baseline_protocol_results_conn.csv (e.g. OCSVM + GuardDog 73.9% clean, 34.9% at 1%; per-group 62.6% at 1%).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/baseline_protocol.py --levels 0 0.01 0.02 0.05 --variants naive --detectors OCSVM PEER IF \
    --feature-sets static+deps+names+GuardDog --boot 100 --tag _ops --save-scores
