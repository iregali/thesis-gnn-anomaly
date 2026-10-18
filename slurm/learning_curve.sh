#!/bin/bash
#SBATCH --job-name=learning_curve
#SBATCH --account=uusei19045
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
# Learning curve (Methodology §33): InfoNCE + GIN with the TUNED training settings
# (results/tuning_choice.csv, as train_tuned.sh) and 10 / 25 / 50 / 100% of the clean
# training packages; depth 0 = no-graph control, depths 1 and 2 = with the graph.
# Dropped packages are removed from the graph entirely.
# The raw-feature curve on the same packages: slurm/learning_curve_raw.sh (CPU).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
FRACS=(0.1 0.25 0.5 1.0)
TAGS=(10 25 50 100)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
read E D LR < <(python - << 'PY'
import sys, pandas as pd
sys.path.insert(0, "src")
from baseline_protocol import RESULTS_DIR
t = pd.read_csv(f"{RESULTS_DIR}/tuning_choice.csv").set_index("loss").loc["infonce"]
print(int(t["epochs"]), int(t["dim"]), t["lr"])
PY
)
echo "InfoNCE + GIN | train fraction $FRAC | tuned: epochs $E dim $D lr $LR"
python src/train_contrastive.py --loss infonce --encoder gin --depths 0 1 2 --levels 0 \
    --rescale all --agg mean --epochs $E --dim $D --lr $LR \
    --train-frac $FRAC --tag _infonce_gin_lc${TAGS[$SLURM_ARRAY_TASK_ID]}
