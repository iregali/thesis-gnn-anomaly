#!/bin/bash
#SBATCH --job-name=download_normal
#SBATCH --partition=staging
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/sample_normal.py --n 15000 --out-root ~/thesis/data/normal_archives
