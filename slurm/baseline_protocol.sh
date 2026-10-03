#!/bin/bash
#SBATCH --job-name=baseline_protocol
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/baseline_protocol.py
