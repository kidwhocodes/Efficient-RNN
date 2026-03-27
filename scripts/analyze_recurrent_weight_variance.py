#!/usr/bin/env python3
"""Plot recurrent weight-size variance vs pruning amount by method.

Uses suite CSV metrics (no model reload):
  - post_rec_weight_abs_std_nz (preferred) -> variance on nonzero recurrent weights
  - fallback: post_rec_weight_abs_std
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALIASES = {
    "fischer": "fisher",
    "random": "random_unstructured",
    "l1": "l1_unstructured",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument(
        "--methods",
        default="noise_prune,l1_unstructured,random_unstructured,obd,fisher",
        help="Comma-separated methods to include.",
    )
    p.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task filters.",
    )
    p.add_argument(
        "--use_all_weights",
        action="store_true",
        help="Use post_rec_weight_abs_std instead of nonzero-only std.",
    )
    p.add_argument("--output_csv", default="")
    p.add_argument("--plot_out", default="")
    return p.parse_args()


def _to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    if x != x:
        return None
    return x


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var ** 0.5


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise SystemExit(f"Missing CSV: {in_path}")

    methods = set()
    for m in args.methods.split(","):
        mm = m.strip().lower()
        if mm:
            methods.add(ALIASES.get(mm, mm))
    task_filters = {t.strip() for t in args.tasks.split(",") if t.strip()}

    std_col = "post_rec_weight_abs_std" if args.use_all_weights else "post_rec_weight_abs_std_nz"
    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)

    with in_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            strategy = (row.get("strategy") or "").strip().lower()
            strategy = ALIASES.get(strategy, strategy)
            if strategy not in methods:
                continue
            task = (row.get("task") or "").strip()
            if task_filters and task not in task_filters:
                continue
            amount = _to_float(row.get("amount", ""))
            std = _to_float(row.get(std_col, ""))
            if amount is None or std is None:
                continue
            grouped[(strategy, amount)].append(std * std)

    if not grouped:
        raise SystemExit("No usable rows found for requested filters/methods.")

    rows_out = []
    print("strategy\tamount\tn\tvar_mean\tvar_std")
    for (strategy, amount) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        vals = grouped[(strategy, amount)]
        vm, vs = _mean_std(vals)
        print(f"{strategy}\t{amount:.1f}\t{len(vals)}\t{vm:.8f}\t{vs:.8f}")
        rows_out.append(
            {
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "n": str(len(vals)),
                "var_mean": f"{vm:.10f}",
                "var_std": f"{vs:.10f}",
            }
        )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["strategy", "amount", "n", "var_mean", "var_std"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote {out}")

    if args.plot_out:
        by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for r in rows_out:
            by_method[r["strategy"]].append((float(r["amount"]), float(r["var_mean"])))
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.5, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel("Amount pruned")
        plt.ylabel("Variance of |W_rec| (post)")
        plt.title("Recurrent weight-size variance vs pruning amount")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
