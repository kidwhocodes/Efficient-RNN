#!/usr/bin/env python3
"""Summarize spectral-radius preservation by pruning amount from a results CSV.

For CTRNN runs, alpha is fixed, so:
    rho_preservation ~= post_alpha_rho / pre_alpha_rho
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


DEFAULT_METHODS = ("noise_prune", "l1_unstructured", "random_unstructured", "obd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Suite results CSV path.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated strategies to include.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional output CSV for grouped summary.",
    )
    parser.add_argument(
        "--per_task",
        action="store_true",
        help="Also print per-task grouped summaries.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task filters (e.g. modcog:ctxdlydm1seqr,modcog:ctxdlydm2seqr).",
    )
    parser.add_argument(
        "--plot_out",
        default="",
        help="Optional output PNG path for rho preservation vs amount plot.",
    )
    return parser.parse_args()


def _to_float(val: str) -> float | None:
    if val is None or val == "":
        return None
    try:
        out = float(val)
    except ValueError:
        return None
    if out != out:  # NaN
        return None
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise SystemExit(f"Missing CSV: {in_path}")

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    task_filters = {t.strip() for t in args.tasks.split(",") if t.strip()}
    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    grouped_task: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)

    with in_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            strat = (row.get("strategy") or "").strip()
            if strat not in methods:
                continue
            task = (row.get("task") or "").strip()
            if task_filters and task not in task_filters:
                continue
            pre_ar = _to_float(row.get("pre_alpha_rho", ""))
            post_ar = _to_float(row.get("post_alpha_rho", ""))
            amount = _to_float(row.get("amount", ""))
            if pre_ar is None or post_ar is None or amount is None:
                continue
            if abs(pre_ar) < 1e-12:
                continue
            preservation = post_ar / pre_ar
            grouped[(strat, amount)].append(preservation)
            grouped_task[(task, strat, amount)].append(preservation)

    if not grouped:
        raise SystemExit("No usable rows found. Check method names and alpha_rho columns.")

    print("strategy\tamount\tn\trho_pres_mean\trho_pres_std")
    rows_out: List[Dict[str, str]] = []
    for (strat, amount) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        vals = grouped[(strat, amount)]
        mean, std = _mean_std(vals)
        print(f"{strat}\t{amount:.1f}\t{len(vals)}\t{mean:.6f}\t{std:.6f}")
        rows_out.append(
            {
                "strategy": strat,
                "amount": f"{amount:.1f}",
                "n": str(len(vals)),
                "rho_pres_mean": f"{mean:.8f}",
                "rho_pres_std": f"{std:.8f}",
            }
        )

    if args.per_task:
        print("\nper-task:")
        print("task\tstrategy\tamount\tn\trho_pres_mean\trho_pres_std")
        for (task, strat, amount) in sorted(grouped_task.keys(), key=lambda x: (x[0], x[1], x[2])):
            vals = grouped_task[(task, strat, amount)]
            mean, std = _mean_std(vals)
            print(f"{task}\t{strat}\t{amount:.1f}\t{len(vals)}\t{mean:.6f}\t{std:.6f}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["strategy", "amount", "n", "rho_pres_mean", "rho_pres_std"],
            )
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote {out_path}")

    if args.plot_out:
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for (strat, amount), vals in grouped.items():
            mean, _ = _mean_std(vals)
            by_method[strat].append((amount, mean))
        plt.figure(figsize=(7.5, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.axhline(1.0, linestyle="--", linewidth=1.0, color="gray")
        plt.xlabel("Amount pruned")
        plt.ylabel("Spectral radius preservation (post/pre)")
        if task_filters:
            plt.title(f"Rho preservation vs pruning ({len(task_filters)} task filter)")
        else:
            plt.title("Rho preservation vs pruning (all selected tasks)")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
