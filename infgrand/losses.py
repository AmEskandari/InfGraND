"""InfGraND influence-weighted supervised + KL distillation losses (paper Eq. 5-9)."""
import torch
import torch.nn.functional as F


def influence_weighted_ce(
    out_log: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    influence: torch.Tensor,
    delta_1: float,
    delta_2: float,
) -> torch.Tensor:
    """L_sup = δ₁ · CE + δ₂ · (CE * I_g[idx]).mean()."""
    ce = F.nll_loss(out_log[idx], labels[idx], reduction="none")
    return delta_1 * ce.mean() + delta_2 * (ce * influence[idx]).mean()


def influence_weighted_kl(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    edge_idx: torch.Tensor,
    influence: torch.Tensor,
    gamma_1: float,
    gamma_2: float,
    tau: float,
) -> torch.Tensor:
    """L_kd = γ₁·KL(s_src || t_dst) + γ₂·(KL * I_g[dst]).mean(), summed over edges."""
    src, dst = edge_idx[0], edge_idx[1]
    log_p = F.log_softmax(logits_student[src] / tau, dim=1)
    q = F.softmax(logits_teacher[dst] / tau, dim=1)
    kl_per_edge = F.kl_div(log_p, q, reduction="none").sum(dim=1)
    weights = influence[dst]
    return gamma_1 * kl_per_edge.mean() + gamma_2 * (kl_per_edge * weights).mean()


def total_loss(L_sup: torch.Tensor, L_kd: torch.Tensor, lamb: float) -> torch.Tensor:
    """L = λ · L_sup + (1 - λ) · L_kd."""
    return lamb * L_sup + (1.0 - lamb) * L_kd
