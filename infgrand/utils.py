"""Reproducibility, graph helpers, and evaluator."""
import random

import dgl
import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_indices(g):
    edge_idx_loop = g.adjacency_matrix(transpose=True)._indices()
    edge_idx_no_loop = dgl.remove_self_loop(g).adjacency_matrix(transpose=True)._indices()
    return (edge_idx_loop, edge_idx_no_loop)


def idx_split(idx, ratio, seed):
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    perm = torch.randperm(n)
    return idx[perm[:cut]], idx[perm[cut:]]


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """Hide a `rate` fraction of test nodes for inductive evaluation.

    Returns obs_idx_{train,val,test}, idx_obs, idx_test_ind. Indices prefixed
    with `obs_` are local to the observed subgraph; bare `idx_` are global.
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)
    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    n_train, n_val = idx_train.shape[0], idx_val.shape[0]
    obs_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_all[:n_train]
    obs_idx_val = obs_all[n_train : n_train + n_val]
    obs_idx_test = obs_all[n_train + n_val :]
    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def get_evaluator():
    def evaluator(out, labels):
        return out.argmax(1).eq(labels).float().mean().item()
    return evaluator
