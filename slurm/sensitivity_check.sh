#!/bin/bash
#SBATCH --job-name=sensitivity
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Sensitivity (mechanism) check: does the encoder ignore features that barely vary among normal packages?
# Clean bundle, 3 seeds, validation set. Outputs results/sensitivity_check{tag}.csv.
# Records every run reported in the thesis (results already in results/sensitivity_check.csv).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate

# 1. Bias across the three losses, default (untuned) settings, GIN depths 0 and 1 (Methodology section 36)
python src/sensitivity_check.py --loss infonce --encoder gin --depths 0 1
python src/sensitivity_check.py --loss triplet --encoder gin --depths 0 1
python src/sensitivity_check.py --loss dgi     --encoder gin --depths 0 1
python src/sensitivity_check.py --loss infonce --encoder gps --depths 1

# 2. Mechanism check reported in the thesis (Results, "Mechanism"): Triplet + GIN depth 1,
#    tuned settings (tuning_choice.csv: 25 epochs, size 128, lr 1e-3), tuned base vs full proposed loss N3
#    (sr_choice.csv: nu 0.02, K 1, weight 2)
COMMON="--loss triplet --encoder gin --depths 1 --epochs 25 --dim 128 --lr 1e-3"
python src/sensitivity_check.py $COMMON --tag _triplet_gin_tuned
python src/sensitivity_check.py $COMMON --strata conn --shift rare --rare-nu 0.02 --shift-k 1 --shift-weight 2 \
    --tag _triplet_gin_n3