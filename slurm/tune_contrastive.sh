#!/bin/bash
#SBATCH --job-name=tune
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/%x_%A_%a.out
# Tuning of the untuned training settings, on VALIDATION only (Methodology §37):
# 3 losses x epochs {25, 50, 100} x embedding size {32, 64, 128} x learning rate {3e-4, 1e-3}
# = 54 tasks; GIN, depth 1, clean training, 3 seeds, small bootstrap (test numbers are not used to choose).
# Selection afterwards: python src/select_tuning.py
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(infonce triplet dgi); EPOCHS=(25 50 100); DIMS=(32 64 128); LRS=(0.0003 0.001)
i=$SLURM_ARRAY_TASK_ID
L=${LOSSES[$((i / 18))]}; r=$((i % 18))
E=${EPOCHS[$((r / 6))]}; r=$((r % 6))
D=${DIMS[$((r / 2))]}; LR=${LRS[$((r % 2))]}
python src/train_contrastive.py --loss $L --encoder gin --depths 1 --levels 0 --rescale all --agg mean \
    --epochs $E --dim $D --lr $LR --boot 20 --tag _tune_${L}_e${E}_d${D}_lr${LR}
