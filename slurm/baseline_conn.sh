#!/bin/bash
#SBATCH --job-name=baseline_conn
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# connectivity subgroups + PEER detector + calibrated combinations (naive variant)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/connectivity.py
python src/baseline_protocol.py --tag _conn
