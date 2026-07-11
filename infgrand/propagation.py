"""Offline multi-hop feature propagation for the student MLP input (paper Eq. 5, 7).

Computes X̃ = POOL({X^{(0)}, X^{(1)}, ..., X^{(P)}}) where X^{(k)} = Ã X^{(k-1)}
and Ã = D̃^{-1/2}(A + I)D̃^{-1/2} is the symmetric-normalized adjacency.
"""
import torch


def _normalized_adjacency(g) -> torch.Tensor:
    """Dense Ã = D̃^{-1/2}(A + I)D̃^{-1/2}."""
    A = g.adjacency_matrix(scipy_fmt="coo").todense()
    A = torch.tensor(A, dtype=torch.float)
    n = A.shape[0]
    A_s = A + torch.eye(n)
    d_inv_sqrt = A_s.sum(dim=1).clamp_min(1e-12).rsqrt()
    return (A_s * d_inv_sqrt.unsqueeze(1)) * d_inv_sqrt.unsqueeze(0)


def propagate_features(g, feats: torch.Tensor, hops: int, pooling: str = "mean") -> torch.Tensor:
    """Return the pooled multi-hop features used as the student MLP input.

    Args:
        g: DGL graph (self-loops will be added via A + I in the normalization).
        feats: (N, D) node features.
        hops: number of propagation steps P. Setting P=0 returns raw features.
        pooling: aggregation over {X^{(0)}, ..., X^{(P)}}. Only "mean" is used
            in the main results; "max" and "min" are supported for the
            paper's aggregation ablation.
    """
    if hops < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")
    if hops == 0:
        return feats

    device = feats.device
    A = _normalized_adjacency(g).to(device)
    stages = [feats]
    for _ in range(hops):
        stages.append(A @ stages[-1])

    stacked = torch.stack(stages, dim=0)
    if pooling == "mean":
        return stacked.mean(dim=0)
    if pooling == "max":
        return stacked.max(dim=0).values
    if pooling == "min":
        return stacked.min(dim=0).values
    raise ValueError(f"Unknown pooling {pooling!r}")
