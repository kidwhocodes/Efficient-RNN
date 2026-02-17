#!/usr/bin/env python3
"""
Plot pruning curves for a single task from a benchmark CSV.

Usage:
    python scripts/plot_single_task_pruning.py \
        --csv results/single_task_pruning_plot.csv \
        --out plots/single_task_pruning/pruning_curve.png \
        --task modcog:ctxdlydm1seql
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pruning curves for one task.")
    parser.add_argument("--csv", required=True, help="Path to the suite results CSV.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path for the plot.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task name to filter (defaults to the first task present).",
    )
    parser.add_argument(
        "--metric",
        default="post_acc",
        help="Metric column to plot on the y-axis (default: post_acc).",
    )
    return parser.parse_args()


def load_data(csv_path: str, task: str | None, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if task:
        df = df[df["task"] == task]
    if df.empty:
        raise ValueError("No rows found for the requested task.")
    if task is None:
        task = df["task"].iloc[0]
        df = df[df["task"] == task]
    df = df[df["strategy"] != "none"].copy()
    df["amount"] = df["amount"].astype(float)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in CSV columns.")
    return df


def plot_curves(df: pd.DataFrame, metric: str, out_path: str) -> str:
    plt.figure(figsize=(8, 5))
    for strategy, group in df.groupby("strategy"):
        group = group.sort_values("amount")
        plt.plot(
            group["amount"],
            group[metric],
            marker="o",
            label=strategy,
        )
    plt.xlabel("Pruning amount (fraction)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Pruning curves for {df['task'].iloc[0]}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def main():
    args = _parse_args()
    df = load_data(args.csv, args.task, args.metric)
    out_path = plot_curves(df, args.metric, args.out)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
