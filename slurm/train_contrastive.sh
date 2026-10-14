#!/bin/bash
#SBATCH --job-name=contrastive
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
# Thesis framework (§3.4-3.6): encoder x contrastive loss -> frozen embeddings -> IF / LOF / OCSVM.
# Base losses with the DEFAULT (untuned) training settings: 50 epochs, size 64, lr 1e-3.
# 3 losses x 5 encoders (GIN, GAT, simplified Transformer "tf", TransformerConv, GraphGPS), levels 0% and 1%, 3 seeds.
# Task index i: loss = LOSSES[i / 5], encoder = ENCODERS[i % 5]. The "tf" encoder is not reported in the thesis.
# The thesis reports the TUNED reruns of these runs: slurm/train_tuned.sh (tags _{loss}_{encoder}_tuned).
# Depth 0 (no graph) does not depend on the encoder: it runs only in the GIN tasks.
# Run one task first as a check:  sbatch --array=0 slurm/train_contrastive.sh
# Other contamination levels (results get a suffix so nothing is overwritten), e.g. GIN only at 2% and 5%:
#   sbatch --array=0,5,10 --export=ALL,LEVELS="0.02 0.05",SUFFIX=_c25 slurm/train_contrastive.sh
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(dgi infonce triplet)
ENCODERS=(gin gat tf tconv gps)
LOSS=${LOSSES[$((SLURM_ARRAY_TASK_ID / 5))]}
ENC=${ENCODERS[$((SLURM_ARRAY_TASK_ID % 5))]}
DEPTHS="1 2"
[ "$ENC" = "gin" ] && DEPTHS="0 1 2"
LEVELS=${LEVELS:-"0 0.01"}
SUFFIX=${SUFFIX:-""}
python src/train_contrastive.py --loss $LOSS --encoder $ENC --depths $DEPTHS --levels $LEVELS \
    --rescale all --agg mean --tag _${LOSS}_${ENC}${SUFFIX}
