#!/bin/bash
#SBATCH --job-name=check_bundles
#SBATCH --partition=rome
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/check_bundles.py
