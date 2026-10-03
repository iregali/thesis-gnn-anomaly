#!/bin/bash
#SBATCH --job-name=release_history
#SBATCH --partition=staging
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/fetch_release_history.py --workers 8
