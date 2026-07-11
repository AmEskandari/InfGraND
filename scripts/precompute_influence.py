"""Precompute influence vectors for all datasets and cache to disk.

Usage:
    python scripts/precompute_influence.py --all
    python scripts/precompute_influence.py --dataset cora --recompute
"""
from __future__ import annotations

import argparse

import torch

from infgrand.data import load_dataset
from infgrand.influence import compute_influence

DATASETS = ["cora", "citeseer", "pubmed",
            "amazon-photo", "coauthor-cs", "coauthor-phy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()

    if not args.all and not args.dataset:
        p.error("pass --dataset NAME or --all")
    targets = DATASETS if args.all else [args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in targets:
        print(f"[{name}] loading ...")
        g, _labels, _it, _iv, _ite = load_dataset(name)
        feats = g.ndata["feat"].to(device)
        g = g.to(device)
        infl = compute_influence(g, feats, name,
                                 cache_dir=args.cache_dir,
                                 recompute=args.recompute)
        print(f"[{name}] N={infl.shape[0]}  mean={infl.mean():.4f}  "
              f"max={infl.max():.4f}  min={infl.min():.4f}")


if __name__ == "__main__":
    main()
