"""CLI entry point: one config -> N seeds -> one CSV row."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from .data import load_dataset
from .influence import compute_influence
from .models import build_student, build_teacher
from .trainer import train_student, train_teacher
from .utils import graph_split, set_seed


def _load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if "extends" in cfg:
        base_path = path.parent / cfg.pop("extends")
        with base_path.open() as f:
            base = yaml.safe_load(f)
        merged = {**base, **cfg}
        return merged
    return cfg


def _run_one_seed(cfg, seed, device, cache_dir):
    set_seed(seed)
    g, labels, idx_train, idx_val, idx_test = load_dataset(cfg["dataset"], seed=seed)
    g = g.to(device)
    feats = g.ndata["feat"].to(device)
    labels = labels.to(device)

    if cfg["exp_setting"] == "tran":
        indices = (idx_train, idx_val, idx_test)
    else:
        indices = graph_split(idx_train, idx_val, idx_test,
                              cfg.get("split_rate", 0.2), seed)

    influence = compute_influence(
        g, feats, cfg["dataset"], cache_dir=cache_dir,
        setting=cfg["exp_setting"],
        n_obs=(len(indices[3]) if cfg["exp_setting"] == "ind" else None),
    )

    # Field-name convention of the published configs: `dropout_s` configures
    # the GNN teacher and `dropout_t` the MLP student.
    teacher = build_teacher(
        cfg["teacher"],
        input_dim=feats.shape[1],
        hidden_dim=cfg["hidden_dim"],
        output_dim=int(labels.max().item()) + 1,
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout_s"],
        num_heads=cfg.get("num_heads", 4),
    ).to(device)
    t_opt = optim.Adam(teacher.parameters(), lr=1e-2,
                       weight_decay=cfg["weight_decay"])
    # Teacher accuracy is reported at the best-validation epoch.
    out_t_all, _, test_teacher, _ = train_teacher(
        cfg, teacher, g, feats, labels, indices, t_opt
    )

    student = build_student(
        input_dim=feats.shape[1],
        hidden_dim=cfg["hidden_dim"],
        output_dim=int(labels.max().item()) + 1,
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout_t"],
    ).to(device)
    s_opt = optim.Adam(student.parameters(), lr=cfg["learning_rate"],
                       weight_decay=cfg["weight_decay"])
    test_acc, test_val, test_best = train_student(
        cfg, student, g, feats, labels, out_t_all, influence, indices, s_opt
    )
    return test_acc, test_val, test_best, test_teacher


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="infgrand.train")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, default=None,
                   help="Override num_seeds in the config.")
    p.add_argument("--max-epoch", type=int, default=None,
                   help="Override max_epoch in the config (useful for smoke tests).")
    p.add_argument("--results-csv", default=None,
                   help="Destination CSV. Default: results/{stem}.csv.")
    p.add_argument("--cache-dir", default="cache",
                   help="Where influence_*.npy files live.")
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    if args.seeds is not None:
        cfg["num_seeds"] = args.seeds
    if args.max_epoch is not None:
        cfg["max_epoch"] = args.max_epoch

    results_csv = Path(args.results_csv) if args.results_csv else (
        Path("results") / f"{Path(args.config).stem}.csv"
    )
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = cfg.get("seed", 2022)
    n_seeds = cfg.get("num_seeds", 5)

    accs, vals, bests, teachers = [], [], [], []
    for i in range(n_seeds):
        s = base_seed + i
        t0 = time.time()
        test_acc, test_val, test_best, test_teacher = _run_one_seed(
            cfg, s, device, args.cache_dir
        )
        accs.append(test_acc)
        vals.append(test_val)
        bests.append(test_best)
        teachers.append(test_teacher)
        print(
            f"seed={s}  test_acc={test_acc:.4f}  test_val={test_val:.4f}  "
            f"test_best={test_best:.4f}  teacher={test_teacher:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

    a = np.array(accs)
    v = np.array(vals)
    b = np.array(bests)
    t = np.array(teachers)
    summary = {
        "test_acc_mean": a.mean(), "test_acc_std": a.std(),
        "test_val_mean": v.mean(), "test_val_std": v.std(),
        "test_best_mean": b.mean(), "test_best_std": b.std(),
        "teacher_mean": t.mean(), "teacher_std": t.std(),
    }
    print("summary:", {k: f"{val:.4f}" for k, val in summary.items()})

    write_header = not results_csv.exists()
    with results_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "dataset", "teacher", "exp_setting",
                "test_acc_mean", "test_acc_std",
                "test_val_mean", "test_val_std",
                "test_best_mean", "test_best_std",
                "teacher_mean", "teacher_std",
                "per_seed_test_acc", "per_seed_test_val",
                "per_seed_test_best", "per_seed_teacher",
                "config_path",
            ])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            cfg["dataset"], cfg["teacher"], cfg["exp_setting"],
            f"{a.mean():.6f}", f"{a.std():.6f}",
            f"{v.mean():.6f}", f"{v.std():.6f}",
            f"{b.mean():.6f}", f"{b.std():.6f}",
            f"{t.mean():.6f}", f"{t.std():.6f}",
            str(accs), str(vals), str(bests), str(teachers),
            str(args.config),
        ])
    print(f"wrote {results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
