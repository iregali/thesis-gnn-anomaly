#!/bin/bash
#SBATCH --job-name=baseline_split
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# dependency-combination check (validation) + per-subgroup detectors (all levels)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/dep_combination_check.py
python src/baseline_protocol.py --variants naive split --detectors IF OCSVM --tag _split
