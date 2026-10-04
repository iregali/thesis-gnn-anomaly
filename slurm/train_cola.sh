#!/bin/bash
#SBATCH --job-name=train_cola
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# plain CoLA under the baseline protocol: all levels, 3 seeds, p2 chosen on validation
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f ~/thesis/data/graph_c00_s0.pt ] || python src/build_graph.py --contam 0 --seed 0
python src/train_cola.py
