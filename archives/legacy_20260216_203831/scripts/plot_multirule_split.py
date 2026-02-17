#!/usr/bin/env python3
"""Plot pre-training vs post-training pruning curves with baselines."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRE_PHASE = {"snip", "synflow", "grasp"}
BASELINE = {"random_unstructured", "l1_unstructured"}


def load_phase_data(csv_path: Path, task: str) -> Dict[str, Dict[str, Dict[float, List[float]]]]:
    data: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        "pre": defaultdict(lambda: defaultdict(list)),
        "post": defaultdict(lambda: defaultdict(list)),
    }
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("task") != task:
                continue
            strategy = row["strategy"]
            amount = float(row["amount"])
            acc = float(row["post_acc"])
            phase = row.get("prune_phase", "post").lower()
            if strategy in BASELINE:
                data["pre"][strategy][amount].append(acc)
                data["post"][strategy][amount].append(acc)
            else:
                if phase not in data:
                    phase = "post"
                data[phase][strategy][amount].append(acc)
    return data


def mean_series(data: Dict[str, Dict[float, List[float]]], strategies) -> Dict[str, List[Tuple[float, float]]]:
    series = {}
    for strat in strategies:
        if strat not in data:
            continue
        points = []
        for amount, vals in sorted(data[strat].items()):
            mean = sum(vals) / len(vals)
            points.append((amount, mean))
        if points:
            series[strat] = points
    return series


def plot_group(series: Dict[str, List[Tuple[float, float]]], title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for strat, pts in sorted(series.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=strat)
    ax.set_title(title)
    ax.set_xlabel("Pruned fraction")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pre/post phase pruning curves with baselines.")
    parser.add_argument("--csv", required=True, help="Results CSV path.")
    parser.add_argument("--task", default="synthetic_multirule", help="Task name to filter.")
    parser.add_argument("--out_dir", default="plots/multirule_split", help="Directory for plots.")
    args = parser.parse_args()

    phase_data = load_phase_data(Path(args.csv), args.task)
    all_strategies = set(phase_data["pre"].keys()) | set(phase_data["post"].keys())
    pre_series = mean_series(phase_data["pre"], PRE_PHASE | BASELINE)
    post_series = mean_series(phase_data["post"], (all_strategies - PRE_PHASE) | BASELINE)

    out_dir = Path(args.out_dir)
    plot_group(pre_series, "Pre-training strategies (with baselines)", out_dir / "pre_phase.png")
    plot_group(post_series, "Post-training strategies (with baselines)", out_dir / "post_phase.png")
    print(f"[done] Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
