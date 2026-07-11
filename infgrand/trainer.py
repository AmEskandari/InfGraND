"""Teacher and student training loops."""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .losses import influence_weighted_ce, influence_weighted_kl, total_loss
from .propagation import propagate_features
from .utils import extract_indices, get_evaluator


@torch.no_grad()
def _eval_gnn(model, g, feats, labels, idx):
    model.eval()
    log_p = F.log_softmax(model(g, feats), dim=1)
    return get_evaluator()(log_p[idx], labels[idx])


@torch.no_grad()
def _eval_mlp(model, feats, labels, idx):
    model.eval()
    log_p = F.log_softmax(model(feats[idx]), dim=1)
    return get_evaluator()(log_p, labels[idx])


def train_teacher(cfg, model, g, feats, labels, indices, optimizer):
    """Train teacher GNN; return (out_t_all, test_acc, test_val, test_best).

    `out_t_all` are the teacher logits over ALL nodes — used to supervise the
    student. In inductive mode we train on the observed subgraph but produce
    logits over the full graph by filling held-out test nodes from a full-graph
    forward pass.
    """
    setting = cfg["exp_setting"]
    if setting == "tran":
        idx_train, idx_val, idx_test = (i.to(feats.device) for i in indices)
    else:
        obs_idx_train, obs_idx_val, _obs_idx_test, idx_obs, idx_test_ind = (
            i.to(feats.device) for i in indices
        )
        obs_g = g.subgraph(idx_obs).to(feats.device)
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]

    best_state, val_best = None, 0.0
    test_val, test_best, patience = 0.0, 0.0, 0
    for epoch in range(1, cfg["max_epoch"] + 1):
        model.train()
        if setting == "tran":
            logits = model(g, feats)
            loss = F.nll_loss(F.log_softmax(logits, dim=1)[idx_train], labels[idx_train])
        else:
            logits = model(obs_g, obs_feats)
            loss = F.nll_loss(F.log_softmax(logits, dim=1)[obs_idx_train], obs_labels[obs_idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if setting == "tran":
            val_acc = _eval_gnn(model, g, feats, labels, idx_val)
            test_acc = _eval_gnn(model, g, feats, labels, idx_test)
        else:
            val_acc = _eval_gnn(model, obs_g, obs_feats, obs_labels, obs_idx_val)
            test_acc = _eval_gnn(model, g, feats, labels, idx_test_ind)

        test_best = max(test_best, test_acc)
        if val_acc >= val_best:
            val_best, test_val = val_acc, test_acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 50:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if setting == "tran":
            out_t_all = model(g, feats)
        else:
            out_t_all = model(g, feats).detach().clone()
            obs_out = model(obs_g, obs_feats)
            out_t_all[idx_obs] = obs_out
    return out_t_all.detach(), test_acc, test_val, test_best


def train_student(cfg, model, g, feats, labels, out_t_all, influence, indices, optimizer):
    """Train student MLP with InfGraND losses; return (test_acc, test_val, test_best).

    The student input is the pooled multi-hop propagation X̃ of the raw features
    (paper Eq. 5, 7), produced once before the loop.
    """
    setting = cfg["exp_setting"]
    if setting == "tran":
        feats_pool = propagate_features(g, feats, hops=cfg["propagation_hops"],
                                        pooling=cfg.get("pooling", "mean"))
        # KD runs over edges including self-loops; the self-loop terms add a
        # per-node direct distillation signal.
        edge_idx_kd = extract_indices(g.to(feats.device))[0].to(feats.device)
        idx_train, idx_val, idx_test = (i.to(feats.device) for i in indices)
        idx_l = idx_train
    else:
        obs_idx_train, obs_idx_val, _obs_idx_test, idx_obs, idx_test_ind = (
            i.to(feats.device) for i in indices
        )
        idx_l = idx_obs[obs_idx_train]
        idx_val_global = idx_obs[obs_idx_val]

        # Inductive protocol: nothing about the hidden test nodes may reach
        # training. Observed rows are propagated over the observed subgraph
        # only; hidden-node rows are filled from full-graph propagation and
        # used exclusively at evaluation time. The KD loss likewise runs over
        # observed-subgraph edges only (mapped back to global node IDs).
        obs_g = g.subgraph(idx_obs)
        feats_pool = torch.zeros_like(feats)
        feats_pool[idx_obs] = propagate_features(
            obs_g, feats[idx_obs], hops=cfg["propagation_hops"],
            pooling=cfg.get("pooling", "mean"))
        full_pool = propagate_features(g, feats, hops=cfg["propagation_hops"],
                                       pooling=cfg.get("pooling", "mean"))
        feats_pool[idx_test_ind] = full_pool[idx_test_ind]

        obs_edges_local = extract_indices(obs_g)[0].to(feats.device)
        edge_idx_kd = torch.stack(
            [idx_obs[obs_edges_local[0]], idx_obs[obs_edges_local[1]]])

    val_best, test_val, test_best, patience = 0.0, 0.0, 0.0, 0
    for epoch in range(1, cfg["max_epoch"] + 1):
        model.train()
        logits = model(feats_pool)
        log_p = F.log_softmax(logits, dim=1)

        L_sup = influence_weighted_ce(log_p, labels, idx_l, influence,
                                      cfg["delta_1"], cfg["delta_2"])
        L_kd = influence_weighted_kl(logits, out_t_all, edge_idx_kd, influence,
                                     cfg["gamma_1"], cfg["gamma_2"], cfg["tau"])
        loss = total_loss(L_sup, L_kd, cfg["lamb"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if setting == "tran":
            val_acc = _eval_mlp(model, feats_pool, labels, idx_val)
            test_acc = _eval_mlp(model, feats_pool, labels, idx_test)
        else:
            val_acc = _eval_mlp(model, feats_pool, labels, idx_val_global)
            test_acc = _eval_mlp(model, feats_pool, labels, idx_test_ind)

        test_best = max(test_best, test_acc)
        if val_acc >= val_best:
            val_best, test_val = val_acc, test_acc
            patience = 0
        else:
            patience += 1
        if patience >= 50:
            break

    return test_acc, test_val, test_best
