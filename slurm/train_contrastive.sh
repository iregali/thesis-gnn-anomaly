#!/bin/bash
#SBATCH --job-name=contrastive
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
# Thesis framework (§3.4-3.6): encoder x contrastive loss -> frozen embeddings -> IF / LOF / OCSVM.
# 3 losses x 3 encoders (GIN, GAT, Graph Transformer), levels 0% and 1%, 3 seeds.
# Depth 0 (no graph) does not depend on the encoder: it runs only in the GIN tasks.
# Run one task first as a check:  sbatch --array=0 slurm/train_contrastive.sh
# Other contamination levels (results get a suffix so nothing is overwritten), e.g. GIN only at 2% and 5%:
#   sbatch --array=0,3,6 --export=ALL,LEVELS="0.02 0.05",SUFFIX=_c25 slurm/train_contrastive.sh
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(dgi infonce triplet)
ENCODERS=(gin gat tf)
LOSS=${LOSSES[$((SLURM_ARRAY_TASK_ID / 3))]}
ENC=${ENCODERS[$((SLURM_ARRAY_TASK_ID % 3))]}
DEPTHS="1 2"
[ "$ENC" = "gin" ] && DEPTHS="0 1 2"
LEVELS=${LEVELS:-"0 0.01"}
SUFFIX=${SUFFIX:-""}
python src/train_contrastive.py --loss $LOSS --encoder $ENC --depths $DEPTHS --levels $LEVELS \
    --rescale all --agg mean --tag _${LOSS}_${ENC}${SUFFIX}
