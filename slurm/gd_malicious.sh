#!/bin/bash
#SBATCH --job-name=gd_malicious
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate

python src/extract_guarddog.py --mode local --out ~/thesis/data/guarddog_malicious.csv
