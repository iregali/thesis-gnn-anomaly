#!/bin/bash
#SBATCH --job-name=ocgnn_rescale
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# one-class GNN with scale_all features (flags standardised), levels 0% and 1%:
# GIN (sum and mean aggregation), GAT, Transformer
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
COMMON="--levels 0 0.01 --rescale all"
python src/train_ocgnn.py $COMMON --encoder gin --agg sum  --depths 0 1 2 --tag _gin_all
python src/train_ocgnn.py $COMMON --encoder gin --agg mean --depths 1 2   --tag _ginmean_all
python src/train_ocgnn.py $COMMON --encoder gat --depths 1 2 --tag _gat_all
python src/train_ocgnn.py $COMMON --encoder tf  --depths 1 2 --tag _tf_all
