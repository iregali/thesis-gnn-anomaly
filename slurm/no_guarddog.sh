#!/bin/bash
#SBATCH --job-name=no_guarddog
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
# Revision (Oct 2026): Triplet + GIN WITHOUT the GuardDog features (--drop-features guarddog),
# to see how much the results and the gain of the proposed negatives depend on GuardDog.
#   task 0: tuned base loss     -> tag _triplet_gin_tuned_nogd
#   task 1: proposed loss (N3)  -> tag _triplet_gin_n3_nogd
# Everything else exactly as train_tuned.sh (task 4) and train_sr.sh (task 24): same tuned training
# settings (tuning_choice.csv) and proposed-loss settings (sr_choice.csv), both chosen WITH GuardDog and
# reused unchanged; depths 0 1 2, levels 0% and 1%, 3 seeds; depth and scorer chosen on validation.
# The rare-feature set of N3 is recomputed without the GuardDog columns: the log lists it.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
L=triplet; ENC=gin
LEVELS=${LEVELS:-"0 0.01"}
read E D LR NU K W < <(python - "$L" << 'EOF'
import sys, pandas as pd
sys.path.insert(0, "src")
from baseline_protocol import RESULTS_DIR
t = pd.read_csv(f"{RESULTS_DIR}/tuning_choice.csv").set_index("loss").loc[sys.argv[1]]
s = pd.read_csv(f"{RESULTS_DIR}/sr_choice.csv").set_index("loss").loc[sys.argv[1]]
print(int(t["epochs"]), int(t["dim"]), t["lr"], s["nu"], int(s["k"]), s["weight"])
EOF
)
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    COND=""; TAG=_${L}_${ENC}_tuned_nogd
else
    COND="--strata conn --shift rare --rare-nu $NU --shift-k $K --shift-weight $W"; TAG=_${L}_${ENC}_n3_nogd
fi
echo "loss $L | encoder $ENC | no GuardDog | ${COND:-base} | tuned: epochs $E dim $D lr $LR | tag $TAG"
python src/train_contrastive.py --loss $L --encoder $ENC --depths 0 1 2 --levels $LEVELS \
    --rescale all --agg mean --epochs $E --dim $D --lr $LR $COND --drop-features guarddog --tag $TAG
