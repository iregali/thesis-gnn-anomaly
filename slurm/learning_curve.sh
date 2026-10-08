#!/bin/bash
#SBATCH --job-name=learning_curve
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
# Learning curve (Methodology §33): InfoNCE + GIN (the configuration chosen on validation at 0%)
# with 10 / 25 / 50 / 100% of the clean training packages; depth 0 = no-graph control,
# depths 1 and 2 = with the graph. Dropped packages are removed from the graph entirely.
# The raw-feature curve on the same packages: slurm/learning_curve_raw.sh (CPU).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
FRACS=(0.1 0.25 0.5 1.0)
TAGS=(10 25 50 100)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
python src/train_contrastive.py --loss infonce --encoder gin --depths 0 1 2 --levels 0 \
    --rescale all --agg mean --train-frac $FRAC --tag _infonce_gin_lc${TAGS[$SLURM_ARRAY_TASK_ID]}
