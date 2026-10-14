#!/bin/bash
# Submit helper, run with:  bash slurm/contamination_c25.sh   (on the login node, NOT with sbatch)
# The 2% and 5% contamination runs reported in the thesis (exploratory analysis, Methodology section 38).
# They reuse the regular job scripts with LEVELS and SUFFIX, so the results get the tag suffix _c25.
cd ~/thesis/thesis-gnn-anomaly
# Base losses with default settings, GIN (tags _{loss}_gin_c25)
sbatch --array=0,5,10 --export=ALL,LEVELS="0.02 0.05",SUFFIX=_c25 slurm/train_contrastive.sh
# Triplet + GIN, tuned base (tag _triplet_gin_tuned_c25)
sbatch --array=4 --export=ALL,LEVELS="0.02 0.05",SUFFIX=_c25 slurm/train_tuned.sh
# Triplet + GIN, full proposed loss N3 (tag _triplet_gin_n3_c25); train_sr.sh task 24 = Triplet, N3, GIN
sbatch --array=24 --export=ALL,LEVELS="0.02 0.05",SUFFIX=_c25 slurm/train_sr.sh
