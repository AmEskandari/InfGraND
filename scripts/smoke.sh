#!/usr/bin/env bash
# Smoke test: 1 seed, 20 epochs, cora/GCN/tran. Should finish in a few minutes.
set -euo pipefail

python -m infgrand.train \
    --config configs/cora_gcn_tran.yaml \
    --seeds 1 \
    --max-epoch 20 \
    --results-csv results/_smoke.csv
