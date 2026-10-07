#!/bin/bash
#SBATCH --job-name=relation_check
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# relations for isolated malware: dependency chains and imitation (analysis only; reads archives as text)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/relation_check.py --workers 16
