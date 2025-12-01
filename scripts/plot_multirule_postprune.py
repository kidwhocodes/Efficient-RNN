#!/usr/bin/env python3
"""Quick plotting helper for pruning results without pandas."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_metric(csv_path: Path, task: str, metric: str, agg_field: str = "seed") -> Dict[str, List[Tuple[float, float]]]:
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
        for row in reader:
            if row.get("task") and row["task"] != task:
                continue
            strategy = row.get("strategy")
            amount_raw = row.get("amount")
            metric_raw = row.get(metric)
            if not strategy or amount_raw is None or metric_raw is None:
                continue
            try:
                amount = float(amount_raw)
                value = float(metric_raw)
            except ValueError:
                continue
            grouped[strategy][amount].append(value)
    aggregated: Dict[str, List[Tuple[float, float]]] = {}
    for strategy, amount_map in grouped.items():
        aggregated[strategy] = []
        for amount, values in sorted(amount_map.items()):
            mean_val = sum(values) / max(1, len(values))
            aggregated[strategy].append((amount, mean_val))
    return aggregated


def plot_curves(data: Dict[str, List[Tuple[float, float]]], out_path: Path, task: str, metric: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    if not data:
        raise ValueError(f"No rows for task '{task}' in CSV.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for strategy, points in sorted(data.items()):
        points = sorted(points, key=lambda x: x[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=strategy)
    ax.set_title(f"{metric} vs sparsity – {task}")
    ax.set_xlabel("Pruned fraction")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pruning accuracy curves without pandas.")
    parser.add_argument("--csv", required=True, help="Input results CSV.")
    parser.add_argument("--task", default="synthetic_multirule", help="Task name to filter.")
    parser.add_argument("--metric", default="post_acc", help="Metric column to plot.")
    parser.add_argument("--out", default="plots/multirule_postprune/accuracy.png", help="Output PNG path.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    data = load_metric(csv_path, args.task, args.metric)
    result = plot_curves(data, out_path, args.task, args.metric)
    print(f"[info] Plot written to {result}")


if __name__ == "__main__":
    main()
