#!/bin/bash
#SBATCH --job-name=ocgnn_transformers
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# one-class GNN with TransformerConv and GraphGPS-style encoders, scale_all features, levels 0% and 1%
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
COMMON="--levels 0 0.01 --rescale all --depths 1 2"
python src/train_ocgnn.py $COMMON --encoder tconv --tag _tconv_all
python src/train_ocgnn.py $COMMON --encoder gps   --tag _gps_all
