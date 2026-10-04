#!/bin/bash
#SBATCH --job-name=train_ocgnn
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
# plain one-class GNN (Deep SVDD), depths 0/1/2, all levels, 3 seeds; GIN, then GAT, then Transformer
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/train_ocgnn.py --encoder gin --tag _gin
python src/train_ocgnn.py --encoder gat --depths 1 2 --tag _gat
python src/train_ocgnn.py --encoder tf --depths 1 2 --tag _tf
