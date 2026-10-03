#!/bin/bash
#SBATCH --job-name=baseline_robust
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# naive + robust + fixed-settings variants, all four feature sets (ablation)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/baseline_protocol.py --variants naive robust fixed \
    --feature-sets static static+GuardDog static+deps+names static+deps+names+GuardDog \
    --tag _robust
