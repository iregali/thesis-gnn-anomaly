#!/bin/bash
#SBATCH --job-name=capacity
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
# Capacity control (thesis Results, "Capacity"): at depth >= 1 the embedding concatenates the release's own
# projection with the graph output, so it is twice as large as at depth 0. This trains the no-graph model
# (depth 0) at the same total size (2 x the tuned size), clean training, 3 seeds.
# Tags: _infonce_d0_dim128, _triplet_d0_dim256, _dgi_d0_dim256.
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
LOSSES=(infonce triplet dgi)
L=${LOSSES[$SLURM_ARRAY_TASK_ID]}
read E D LR < <(python - "$L" << 'PY'
import sys, pandas as pd
sys.path.insert(0, "src")
from baseline_protocol import RESULTS_DIR
t = pd.read_csv(f"{RESULTS_DIR}/tuning_choice.csv").set_index("loss").loc[sys.argv[1]]
print(int(t["epochs"]), int(t["dim"]), t["lr"])
PY
)
D2=$((2 * D))
echo "loss $L | depth 0 | epochs $E | size $D2 (2 x $D) | lr $LR"
python src/train_contrastive.py --loss $L --encoder gin --depths 0 --levels 0 \
    --rescale all --agg mean --epochs $E --dim $D2 --lr $LR --tag _${L}_d0_dim${D2}
