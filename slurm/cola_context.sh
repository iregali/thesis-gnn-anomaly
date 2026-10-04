#!/bin/bash
#SBATCH --job-name=cola_context
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# CoLA + diagnostics with context nodes described by their training users (leave-one-out)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/train_cola.py --context-features users --tag _ctx
python src/diagnose_cola.py --context-features users --tag _ctx
