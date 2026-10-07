#!/bin/bash
#SBATCH --job-name=twins_vs_detector
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# near-copies among malware the OCSVM misses vs catches (validation; analysis only)
# needs data/artifacts_cache.pkl from slurm/artifact_twins.sh
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/twins_vs_detector_check.py --level 0.0 --seeds 0
python src/twins_vs_detector_check.py --level 0.01 --seeds 0 1 2
