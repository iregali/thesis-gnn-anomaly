#!/bin/bash
#SBATCH --job-name=build_graph
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/make_campaign_splits.py --contam-share 0.25
for c in 0.01 0.02 0.05; do
  for s in 0 1 2; do
    python src/build_graph.py --contam $c --seed $s
  done
done
