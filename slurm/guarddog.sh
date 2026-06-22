#!/bin/bash
#SBATCH --job-name=guarddog
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=guarddog_%j.log

module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/extract_guarddog.py