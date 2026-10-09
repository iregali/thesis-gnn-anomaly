#!/bin/bash
#SBATCH --job-name=tune_sr
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-41
#SBATCH --output=logs/%x_%A_%a.out
# Settings of the proposed loss (Methodology section 38), chosen on VALIDATION only, full loss (N3):
#   InfoNCE, Triplet: nu {0.02, 0.05, 0.10} x copies K {1, 4} x weight {0.5, 1, 2} = 18 each (tasks 0-35)
#   DGI: nu x K = 6 (tasks 36-41; the weight is not used by DGI)
# GIN depth 1, clean training, 3 seeds, small bootstrap; one feature changed per copy.
# Training settings (epochs, embedding size, lr) = the section 37 choice in tuning_choice.csv.
# Selection afterwards: python src/select_sr.py
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
NUS=(0.02 0.05 0.10); KS=(1 4); WS=(0.5 1 2)
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 36 ]; then
    LOSSES=(infonce triplet); L=${LOSSES[$((i / 18))]}; r=$((i % 18))
    NU=${NUS[$((r / 6))]}; r=$((r % 6)); K=${KS[$((r / 3))]}; W=${WS[$((r % 3))]}
else
    L=dgi; r=$((i - 36)); NU=${NUS[$((r / 2))]}; K=${KS[$((r % 2))]}; W=1
fi
read E D LR < <(python - "$L" << 'EOF'
import sys, pandas as pd
sys.path.insert(0, "src")
from baseline_protocol import RESULTS_DIR
t = pd.read_csv(f"{RESULTS_DIR}/tuning_choice.csv").set_index("loss").loc[sys.argv[1]]
print(int(t["epochs"]), int(t["dim"]), t["lr"])
EOF
)
echo "loss $L | nu $NU | K $K | weight $W | tuned: epochs $E dim $D lr $LR"
python src/train_contrastive.py --loss $L --encoder gin --depths 1 --levels 0 --rescale all --agg mean \
    --epochs $E --dim $D --lr $LR --strata conn --shift rare --rare-nu $NU --shift-k $K --shift-weight $W \
    --boot 20 --tag _srtune_${L}_nu${NU}_k${K}_w${W}
