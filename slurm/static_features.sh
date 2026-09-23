#!/bin/bash
#SBATCH --job-name=static_features
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/extract_static_features.py --root ~/thesis/pypi_malregistry --out ~/thesis/data/static_malicious.csv
python src/extract_static_features.py --root ~/thesis/data/normal_archives --out ~/thesis/data/static_normal.csv
