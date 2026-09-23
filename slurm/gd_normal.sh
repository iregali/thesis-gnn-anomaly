#!/bin/bash
#SBATCH --job-name=gd_normal
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/extract_guarddog.py --mode local --local-root ~/thesis/data/normal_archives --out ~/thesis/data/guarddog_normal.csv
