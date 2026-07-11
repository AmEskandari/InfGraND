# InfGraND: Influence-Guided GNN-to-MLP Knowledge Distillation

[![arXiv](https://img.shields.io/badge/arXiv-2601.08033-b31b1b.svg)](https://arxiv.org/abs/2601.08033)
[![TMLR](https://img.shields.io/badge/TMLR-2026-blue.svg)](https://openreview.net/forum?id=lfzHR3YwlD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-1.1+-orange.svg)](https://www.dgl.ai/)

Reference implementation for our **TMLR** paper. InfGraND distills a pretrained
GNN teacher into an MLP student using two influence-weighted losses, where
per-node influence `I_g(v)` measures how much a node propagates information
through the graph (Algorithm 1 in the paper).

This is the official implementation. It **reproduces the paper's main results**
(Table 1) — 6 datasets × 3 teachers × {transductive, inductive} = 36 cells,
averaged over 5 seeds — from per-cell config files, and includes the influence
computation, both influence-weighted losses, and the ablation switches. See
Acknowledgement for prior work this implementation draws on.

## Installation

```bash
pip install -r requirements.txt   # PyTorch 2.0+, DGL 1.1+
pip install -e .                  # makes `python -m infgrand.train` work
```

InfGraND targets modern DGL / PyTorch versions.

## Quick start

```bash
# Smoke test (~3 minutes on GPU, ~15 on CPU):
bash scripts/smoke.sh

# A single Table 1 cell, 5 seeds, full 500 epochs:
python -m infgrand.train --config configs/cora_gcn_tran.yaml
# -> results/cora_gcn_tran.csv with mean ± std and per-seed values
```

## Reproducing Table 1

```bash
bash scripts/reproduce_table1.sh
# loops the 36 configs and writes results/{name}.csv
# then aggregates into results/table1.md
```

Wall-clock: ~6–12 GPU-hours on a single A100. Precomputed influence vectors
in `cache/` ship with the repo, so a reproduction run requires no
precomputation. To regenerate from scratch:

```bash
python scripts/precompute_influence.py --all --recompute
```

## How it works

`I_g(v)` is built from the cosine similarity between raw features `x_i` and
2-hop propagated features `(Ã²x)_j` (see [infgrand/influence.py](infgrand/influence.py)).

Training combines two losses (see [infgrand/losses.py](infgrand/losses.py)):

- `L_sup = δ₁·CE + δ₂·(CE · I_g[idx]).mean()` — influence-weighted supervision
- `L_kd  = γ₁·KL(s_src ‖ t_dst) + γ₂·(KL · I_g[dst]).mean()` — influence-weighted distillation
- `L = λ·L_sup + (1-λ)·L_kd`

The student MLP receives **propagated** features `X̃ = mean({X^{(0)}, …, X^{(P)}})`
where `P` (the `propagation_hops` field in each YAML) varies per
(dataset, teacher) — see `configs/*.yaml`.

## Ablations without code changes

The paper's ablations on the influence component can be reproduced via HP
overrides on the same configs:

- **w/o Influence** — set `delta_2: 0` and `gamma_2: 0` in the YAML
- **w/o Propagation** — set `propagation_hops: 0` in the YAML

## Citation

```bibtex
@article{
eskandari2026infgrand,
title={InfGra{ND}: An Influence-Guided {GNN}-to-{MLP} Knowledge Distillation},
author={Amir Eskandari and Aman Anand and Elyas Rashno and Farhana Zulkernine},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=lfzHR3YwlD}
}
```

## Acknowledgement

We thank the authors of prior open-source GNN-to-MLP distillation work, whose
public implementations informed parts of our data loading and evaluation
scaffolding. The InfGraND-specific contributions are the influence-weighting of
both the supervised and KD losses and the offline multi-hop feature propagation
used as the student input.
