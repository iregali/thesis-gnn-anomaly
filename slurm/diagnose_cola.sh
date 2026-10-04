#!/bin/bash
#SBATCH --job-name=diagnose_cola
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
# CoLA diagnostics: degree dependence + contextual-anomaly injection on the real graph
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/diagnose_cola.py
