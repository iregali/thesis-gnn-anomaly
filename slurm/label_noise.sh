#!/bin/bash
#SBATCH --job-name=label_noise
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Label-noise check (Methodology section 15 open item): proof-of-concept / research probes, CTF material and
# test uploads among the malicious packages, and whether detectors catch them differently.
# Archives are read as text only (never extracted, installed or executed).
source ~/setup_env.sh
cd ~/thesis/thesis-gnn-anomaly
source venv/bin/activate
python src/label_noise_check.py
