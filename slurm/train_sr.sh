#!/bin/bash
#SBATCH --job-name=train_sr
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-47
#SBATCH --output=logs/%x_%A_%a.out
# Proposed loss, ablation conditions of thesis section 3.7.3 (Methodology section 38):
#   N1 same-group negatives only | N2 rarity-shifted negatives only | N3 both (full) | N4 both, uniform shift
# 3 losses x 4 conditions x 4 encoders (GIN, GAT, TransformerConv, GraphGPS) = 48 tasks; levels 0% and 1%, 3 seeds.
# Depth 0 (no graph) runs only in the GIN tasks, as in train_contrastive.sh.
# Settings: tuning_choice.csv (epochs, dim, lr; section 37) and sr_choice.csv (nu, K, weight; select_sr.py).
# Tags: _{loss}_{encoder}_n{1..4}, the names used in config/primary_comparisons.json.
# Run one task first as a check:  sbatch --array=8 slurm/train_sr.sh   (infonce, N3, GIN)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(infonce triplet dgi); ENCODERS=(gin gat tconv gps)
i=$SLURM_ARRAY_TASK_ID
L=${LOSSES[$((i / 16))]}; r=$((i % 16))
C=$((r / 4 + 1)); ENC=${ENCODERS[$((r % 4))]}
DEPTHS="1 2"
[ "$ENC" = "gin" ] && DEPTHS="0 1 2"
LEVELS=${LEVELS:-"0 0.01"}
SUFFIX=${SUFFIX:-""}
read E D LR NU K W < <(python - "$L" << 'EOF'
import sys, pandas as pd
sys.path.insert(0, "src")
from baseline_protocol import RESULTS_DIR
t = pd.read_csv(f"{RESULTS_DIR}/tuning_choice.csv").set_index("loss").loc[sys.argv[1]]
s = pd.read_csv(f"{RESULTS_DIR}/sr_choice.csv").set_index("loss").loc[sys.argv[1]]
print(int(t["epochs"]), int(t["dim"]), t["lr"], s["nu"], int(s["k"]), s["weight"])
EOF
)
case $C in
    1) COND="--strata conn" ;;
    2) COND="--shift rare --rare-nu $NU --shift-k $K --shift-weight $W" ;;
    3) COND="--strata conn --shift rare --rare-nu $NU --shift-k $K --shift-weight $W" ;;
    4) COND="--strata conn --shift uniform --shift-k $K --shift-weight $W" ;;
esac
echo "loss $L | encoder $ENC | N$C: $COND | tuned: epochs $E dim $D lr $LR"
python src/train_contrastive.py --loss $L --encoder $ENC --depths $DEPTHS --levels $LEVELS \
    --rescale all --agg mean --epochs $E --dim $D --lr $LR $COND --tag _${L}_${ENC}_n${C}${SUFFIX}
