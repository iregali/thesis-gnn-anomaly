#!/bin/bash
#SBATCH --job-name=context_check
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
# injection test with context profiles only for dependencies with >= k training users
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
for k in 3 10; do
  python src/diagnose_cola.py --skip-degree --context-features users --context-min-users $k
done