#!/bin/bash
#SBATCH --job-name=guarddog_run
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=guarddog_run.out
cd $HOME/thesis-gnn-anomaly

mkdir -p results

find ~/thesis/pypi_malregistry -name "*.tar.gz" \
| xargs -P 8 -I {} bash -c '
  file="{}"
  name=$(basename "$file" .tar.gz)

  # skip if already done
  if [ -f "results/${name}.json" ]; then
    exit 0
  fi

  guarddog pypi scan --output-format json "$file" \
    > "results/${name}.json"
'