#!/bin/bash
#SBATCH --job-name=learning_curve_raw
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Raw-feature learning curve (OCSVM, Isolation Forest) on the same packages as slurm/learning_curve.sh
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/learning_curve_raw.py
