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
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(dgi infonce triplet)
ENCODERS=(gin gat tf)
LOSS=${LOSSES[$((SLURM_ARRAY_TASK_ID / 3))]}
ENC=${ENCODERS[$((SLURM_ARRAY_TASK_ID % 3))]}
DEPTHS="1 2"
[ "$ENC" = "gin" ] && DEPTHS="0 1 2"
python src/train_contrastive.py --loss $LOSS --encoder $ENC --depths $DEPTHS --levels 0 0.01 --rescale all --agg mean
