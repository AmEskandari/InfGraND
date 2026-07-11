"""Read results/{name}.csv files and emit a Markdown Table 1 view."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DATASETS = ["cora", "citeseer", "pubmed",
            "amazon-photo", "coauthor-cs", "coauthor-phy"]
TEACHERS = ["GCN", "GAT", "SAGE"]


def _latest_row(path: Path) -> dict | None:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="indir", default="results")
    p.add_argument("--out", default="results/table1.md")
    args = p.parse_args()

    indir = Path(args.indir)
    cells = defaultdict(lambda: defaultdict(dict))  # cells[setting][dataset][teacher]
    for csv_path in indir.glob("*.csv"):
        row = _latest_row(csv_path)
        if not row:
            continue
        ds = row.get("dataset")
        t = row.get("teacher")
        s = row.get("exp_setting")
        if ds not in DATASETS or t not in TEACHERS or s not in {"tran", "ind"}:
            continue
        mean = float(row["test_val_mean"]) * 100
        std = float(row["test_val_std"]) * 100
        cells[s][ds][t] = (mean, std)

    out = Path(args.out)
    lines = ["# Reproduction of paper Table 1\n"]
    for setting, title in [("tran", "Transductive"), ("ind", "Inductive")]:
        lines.append(f"## {title}\n")
        header = "| Dataset | " + " | ".join(TEACHERS) + " |"
        sep = "|---" * (len(TEACHERS) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for ds in DATASETS:
            row_cells = []
            for t in TEACHERS:
                cell = cells[setting].get(ds, {}).get(t)
                row_cells.append(f"{cell[0]:.2f} ± {cell[1]:.2f}" if cell else "—")
            lines.append(f"| {ds} | " + " | ".join(row_cells) + " |")
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
