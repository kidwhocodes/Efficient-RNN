#!/usr/bin/env python3
"""Batch analysis for pruning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_cache = ROOT / ".matplotlib_cache"
mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctrnn_training.analysis.aggregators import (
    load_experiment_records,
    load_metrics_jsons,
    pairwise_deltas,
    stats_by_strategy,
    paired_ttest_vs_baseline,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        print(f"[warn] No data to write for {path.name}")
        return
    df.to_csv(path, index=False)
    print(f"[info] Wrote {len(df)} rows -> {path}")


def plot_accuracy(summary: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    tasks = summary["task"].unique()
    for task in tasks:
        subset = summary[summary["task"] == task]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for strategy, grp in subset.groupby("strategy"):
            ax.errorbar(
                grp["amount"],
                grp["post_acc_mean"],
                yerr=grp.get("post_acc_std", 0.0),
                marker="o",
                label=strategy,
            )
        ax.set_title(f"Accuracy vs Sparsity – {task}")
        ax.set_xlabel("Pruned Fraction")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize="small")
        fig.tight_layout()
        fig_path = out_dir / f"accuracy_{task.replace(':', '_')}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[info] Plot -> {fig_path}")


def structural_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    keys = [col for col in ("task", "strategy", "amount") if col in metrics_df.columns]
    if not keys:
        return pd.DataFrame()
    metrics_cols = [
        col
        for col in metrics_df.columns
        if col not in {"run_dir"}
        and metrics_df[col].dtype != object
    ]
    grouped = (
        metrics_df.groupby(keys)[metrics_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pruning suite results")
    parser.add_argument("--csv", default="results/neurosuite_pruning.csv")
    parser.add_argument("--metrics_dir", default="results")
    parser.add_argument("--out", default="analysis_outputs")
    parser.add_argument("--baseline", default="noise_prune")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    df = load_experiment_records(args.csv)
    print(f"[info] Loaded {len(df)} experiment rows from {args.csv}")

    summary = stats_by_strategy(df)
    save_dataframe(summary, out_dir / "summary_by_strategy.csv")

    deltas = pairwise_deltas(df, baseline=args.baseline)
    save_dataframe(deltas, out_dir / "delta_vs_baseline.csv")

    try:
        ttests = paired_ttest_vs_baseline(df, metric="post_acc", baseline=args.baseline)
        save_dataframe(ttests, out_dir / "ttests_vs_baseline.csv")
    except ImportError:
        print("[warn] scipy not available; skipping t-tests")

    plot_accuracy(summary, out_dir / "plots")

    metrics_df = load_metrics_jsons(args.metrics_dir)
    save_dataframe(metrics_df, out_dir / "metrics_flat.csv")
    structural = structural_summary(metrics_df)
    save_dataframe(structural, out_dir / "structural_summary.csv")

    print("[done] Analysis complete")


if __name__ == "__main__":
    main()
