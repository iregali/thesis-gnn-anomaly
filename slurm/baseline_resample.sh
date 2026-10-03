#!/bin/bash
#SBATCH --job-name=baseline_resample
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out
# normal-resampling check: 5 different assignments of the normal packages (malware unchanged)
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/baseline_protocol.py --levels 0 0.01 --normal-salt ${SLURM_ARRAY_TASK_ID} --boot 200
