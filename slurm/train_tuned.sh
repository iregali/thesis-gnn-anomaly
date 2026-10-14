#!/bin/bash
#SBATCH --job-name=train_tuned
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
# Base losses with the TUNED training settings (Methodology section 37; thesis Section "Shared training configuration").
# These are the base-loss results reported in the thesis and the "tuned base" side of the pre-registered comparisons.
# 3 losses x 4 encoders (GIN, GAT, TransformerConv, GraphGPS) = 12 tasks; levels 0% and 1%, 3 seeds.
# Task index i: loss = LOSSES[i / 4], encoder = ENCODERS[i % 4]. Depth 0 (no graph) runs only in the GIN tasks.
# Settings per loss from results/tuning_choice.csv (select_tuning.py, chosen on validation with GIN depth 1):
#   InfoNCE 100 epochs / size 64 / lr 1e-3; Triplet 25 / 128 / 1e-3; DGI 100 / 128 / 1e-3.
# Other contamination levels get a suffix, e.g. Triplet + GIN at 2% and 5% (tag _triplet_gin_tuned_c25):
#   sbatch --array=4 --export=ALL,LEVELS="0.02 0.05",SUFFIX=_c25 slurm/train_tuned.sh
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(infonce triplet dgi); ENCODERS=(gin gat tconv gps)
i=$SLURM_ARRAY_TASK_ID
L=${LOSSES[$((i / 4))]}; ENC=${ENCODERS[$((i % 4))]}
DEPTHS="1 2"
[ "$ENC" = "gin" ] && DEPTHS="0 1 2"
LEVELS=${LEVELS:-"0 0.01"}
SUFFIX=${SUFFIX:-""}
read E D LR < <(python - "$L" << 'PY'
import sys, pandas as pd
sys.path.insert(0, "src")
from baseline_protocol import RESULTS_DIR
t = pd.read_csv(f"{RESULTS_DIR}/tuning_choice.csv").set_index("loss").loc[sys.argv[1]]
print(int(t["epochs"]), int(t["dim"]), t["lr"])
PY
)
echo "loss $L | encoder $ENC | tuned: epochs $E dim $D lr $LR | levels $LEVELS"
python src/train_contrastive.py --loss $L --encoder $ENC --depths $DEPTHS --levels $LEVELS \
    --rescale all --agg mean --epochs $E --dim $D --lr $LR --tag _${L}_${ENC}_tuned${SUFFIX}
