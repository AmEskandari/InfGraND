"""Per-node influence score I_g(v) (paper Algorithm 1, Eq. 3-4).

I_g(v_i) is derived from the cosine similarity between x_i and (Ã² x)_j over all
j: we compute the N×N similarity matrix S in a row-blocked GPU matmul, min-max
normalize it globally, sum |S| along rows, and divide by the max — producing an
N-dimensional float32 vector.

The N×N similarity is never materialized: it is streamed in row blocks, and the
resulting vector (a few kB to ~140 kB per dataset) is cached on disk so training
runs are deterministic and need no precomputation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .propagation import _normalized_adjacency


def _aax(g, feats: torch.Tensor) -> torch.Tensor:
    A = _normalized_adjacency(g).to(feats.device)
    return A @ (A @ feats)


def _cosine_influence(X: torch.Tensor, Y: torch.Tensor, block_size: int = 4096,
                      col_limit: int | None = None) -> torch.Tensor:
    """I_g via row-blocked cosine similarity between X[i] and Y[j].

    Mirrors the published computation in two passes (one to find global min/max,
    one to accumulate row sums after min-max normalization), keeping peak memory
    proportional to block_size * N rather than N².

    `col_limit` implements the inductive processing: the min/max normalization
    is computed over the full N×N matrix, while the row sums run over only the
    first `col_limit` columns, keeping held-out test columns out of the
    training weights.
    """
    eps = 1e-12
    X_n = X / X.norm(dim=1, keepdim=True).clamp_min(eps)
    Y_n = Y / Y.norm(dim=1, keepdim=True).clamp_min(eps)
    n = X_n.shape[0]

    s_min = float("inf")
    s_max = float("-inf")
    for start in range(0, n, block_size):
        block = X_n[start : start + block_size] @ Y_n.T
        s_min = min(s_min, block.min().item())
        s_max = max(s_max, block.max().item())
    rng = max(s_max - s_min, eps)

    Y_sum = Y_n if col_limit is None else Y_n[:col_limit]
    row_sum = torch.empty(n, device=X.device)
    for start in range(0, n, block_size):
        block = X_n[start : start + block_size] @ Y_sum.T
        block = (block - s_min) / rng
        row_sum[start : start + block.shape[0]] = block.abs().sum(dim=1)

    return row_sum / row_sum.max().clamp_min(eps)


def compute_influence(
    g,
    feats: torch.Tensor,
    dataset: str,
    cache_dir: str | os.PathLike = "cache",
    recompute: bool = False,
    block_size: int = 4096,
    setting: str = "tran",
    n_obs: int | None = None,
) -> torch.Tensor:
    """Return I_g as an (N,) tensor on `feats.device`; cache to disk.

    In the inductive setting the row sums cover only the first `n_obs`
    similarity columns. `n_obs` is deterministic per dataset
    (train + val + retained test), so the vector is cached per setting.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if setting == "ind":
        if n_obs is None:
            raise ValueError("n_obs is required when setting='ind'")
        path = cache_dir / f"influence_{dataset}_ind{n_obs}.npy"
        col_limit = n_obs
    else:
        path = cache_dir / f"influence_{dataset}.npy"
        col_limit = None

    if path.exists() and not recompute:
        arr = np.load(path)
        return torch.from_numpy(arr).to(feats.device)

    with torch.no_grad():
        feats_cpu = feats.detach().to(torch.float32)
        AAX = _aax(g, feats_cpu)
        infl = _cosine_influence(feats_cpu, AAX, block_size=block_size,
                                 col_limit=col_limit)

    np.save(path, infl.detach().cpu().numpy().astype(np.float32))
    return infl.to(feats.device)
