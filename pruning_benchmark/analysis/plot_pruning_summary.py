"""Plot mean accuracy curves aggregated across tasks."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


def compute_means(rows, metric: str, group_field: str, amount_field: str, filters: Dict[str, str] | None):
    agg: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if filters and any(row.get(k) != v for k, v in filters.items()):
            continue
        group = row.get(group_field)
        try:
            amount = float(row.get(amount_field, "nan"))
            value = float(row.get(metric, "nan"))
        except (TypeError, ValueError):
            continue
        if group is None or amount != amount or value != value:
            continue
        agg[group][amount].append(value)
    return agg


def plot_curves(agg, metric: str, amount_field: str, output_path: str):
    if not agg:
        raise ValueError("No data available to plot.")
    plt.figure(figsize=(8, 5))
    for group in sorted(agg):
        pts = sorted(agg[group].items())
        xs = [a for a, _ in pts]
        ys = [sum(vals) / len(vals) for _, vals in pts]
        plt.plot(xs, ys, marker="o", label=group)
    plt.xlabel(amount_field)
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot average pruning performance across tasks.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--metric", default="post_acc_sequence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group_field", default="strategy")
    parser.add_argument("--amount_field", default="amount")
    parser.add_argument("--filter_task", default=None, help="Optional specific task to plot (otherwise aggregate all).")
    args = parser.parse_args()

    with open(args.input_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    filters = {"task": args.filter_task} if args.filter_task else None
    agg = compute_means(rows, args.metric, args.group_field, args.amount_field, filters)
    plot_curves(agg, args.metric, args.amount_field, args.output)
    print(f"[done] Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
