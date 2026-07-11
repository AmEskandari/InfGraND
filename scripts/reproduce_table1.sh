#!/usr/bin/env bash
# Reproduce paper Table 1: 6 datasets x 3 teachers x {tran, ind} = 36 cells.
# Run from the repo root.
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-cache}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${RESULTS_DIR}"

for cfg in configs/*_tran.yaml configs/*_ind.yaml; do
    name=$(basename "${cfg}" .yaml)
    out="${RESULTS_DIR}/${name}.csv"
    echo "=== ${name} ==="
    python -m infgrand.train --config "${cfg}" \
        --results-csv "${out}" \
        --cache-dir "${CACHE_DIR}"
done

python scripts/aggregate_results.py --in "${RESULTS_DIR}" --out "${RESULTS_DIR}/table1.md"
echo "wrote ${RESULTS_DIR}/table1.md"
