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