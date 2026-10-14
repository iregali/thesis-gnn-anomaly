#!/bin/bash
#SBATCH --job-name=aug_checks
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
# Masking and temperature tests (Methodology section 35): does feature masking, or the uniformity of InfoNCE
# (temperature), explain why contrastive embeddings trail the raw features? InfoNCE, default (untuned) settings,
# clean training, 3 seeds. Main run = mask 0.2, tau 0.5 (tags _infonce_{encoder}).
# 3 encoders x 6 variants = 18 tasks; GIN at depths 0 and 1, TransformerConv and GraphGPS at depth 1.
# Tags: _infonce_{encoder}_mask{00,10,40} and _infonce_{encoder}_tau{010,100,200}.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
ENCODERS=(gin tconv gps)
FLAGS=("--mask-p 0.0" "--mask-p 0.1" "--mask-p 0.4" "--tau 0.1" "--tau 1.0" "--tau 2.0")
TAGS=(mask00 mask10 mask40 tau010 tau100 tau200)
i=$SLURM_ARRAY_TASK_ID
ENC=${ENCODERS[$((i / 6))]}; V=$((i % 6))
DEPTHS="1"
[ "$ENC" = "gin" ] && DEPTHS="0 1"
echo "InfoNCE | encoder $ENC | ${FLAGS[$V]}"
python src/train_contrastive.py --loss infonce --encoder $ENC --depths $DEPTHS --levels 0 \
    --rescale all --agg mean ${FLAGS[$V]} --tag _infonce_${ENC}_${TAGS[$V]}
