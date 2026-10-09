#!/bin/bash
#SBATCH --job-name=sensitivity
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Point 1 diagnostic: does the encoder ignore features that barely vary among normal packages?
# InfoNCE + GIN (depth 0 and 1) and GraphGPS (depth 1), clean bundle, 3 seeds.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/sensitivity_check.py --loss infonce --encoder gin --depths 0 1
python src/sensitivity_check.py --loss infonce --encoder gps --depths 1
