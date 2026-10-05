#!/bin/bash
#SBATCH --job-name=artifact_twins
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# density-matched shared-artifact twin check (analysis only; reads archives as text)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/artifact_twins_check.py --workers 16
