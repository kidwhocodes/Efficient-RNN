#!/usr/bin/env python3
"""
Generate standardized summary tables and plots for pruning suites.

This script ingests a CSV exported by `run_suite_from_config`, normalizes key
columns (task, strategy, amount), aggregates metrics, and writes:
  - results/<prefix>_summary.csv with mean/std/count statistics
  - plots/<prefix>_* plots showing post accuracy / delta accuracy vs sparsity
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_METRICS = ("post_acc", "post_loss", "delta_post_acc")
STRATEGY_LABELS = {
    "random_unstructured": "Random Unstructured",
    "l1_unstructured": "L1 Magnitude",
    "noise_prune": "Noise Prune",
    "woodfisher": "WoodFisher",
    "causal": "Causal Neuron Prune",
    "set": "SET Rewire",
}


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in required:
        if col not in df.columns:
            if col == "delta_post_acc":
                df[col] = df.get("pre_acc", np.nan) - df.get("post_acc", np.nan)
            else:
                df[col] = np.nan
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["task"] = df["task"].astype(str).str.strip()
    df["task"] = df["task"].str.replace("modcog:", "", regex=False)
    df["strategy"] = df["strategy"].astype(str).str.strip()
    df["strategy_label"] = df["strategy"].map(lambda name: STRATEGY_LABELS.get(name, name))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    return df


def summarise(df: pd.DataFrame, group_fields: Tuple[str, ...], metrics: Tuple[str, ...]) -> pd.DataFrame:
    aggregates = {}
    for metric in metrics:
        aggregates[metric] = ["mean", "std", "count"]
    grouped = df.groupby(list(group_fields)).agg(aggregates).reset_index()
    grouped.columns = ["_".join(filter(None, col)).strip("_") for col in grouped.columns]
    return grouped


def plot_task_curves(
    summary: pd.DataFrame,
    metric: str,
    *,
    task_field: str,
    strategy_field: str,
    amount_field: str,
    amount_values: Tuple[float, ...] | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, task_rows in summary.groupby(task_field):
        plt.figure(figsize=(8, 4.5))
        for strategy, strat_rows in task_rows.groupby(strategy_field):
            if strategy == "none":
                continue
            rows = strat_rows.copy()
            if amount_values is not None:
                rows = rows[rows[amount_field].isin(amount_values)]
            if rows.empty:
                continue
            rows.sort_values(amount_field, inplace=True)
            means = rows[f"{metric}_mean"].to_numpy()
            stds = rows.get(f"{metric}_std", 0.0)
            stds = stds.fillna(0.0) if hasattr(stds, "fillna") else stds
            amounts = rows[amount_field].to_numpy()
            if len(amounts) == 0:
                continue
            label = rows["strategy_label"].iloc[0] if "strategy_label" in rows else strategy
            plt.errorbar(
                amounts,
                means,
                yerr=stds,
                fmt="-o",
                capsize=3,
                label=label,
            )
        plt.xlabel("prune_amount")
        plt.ylabel(f"{metric} (mean)")
        plt.title(f"{task} — {metric}")
        if metric == "delta_post_acc":
            plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        # auto-scale y range for clarity
        all_rows = summary[(summary[task_field] == task) & (summary[strategy_field] != "none")]
        if not all_rows.empty:
            vals = all_rows[f"{metric}_mean"].dropna()
            if not vals.empty:
                vmin, vmax = vals.min(), vals.max()
                if np.isfinite(vmin) and np.isfinite(vmax):
                    pad = max(0.01, 0.1 * (vmax - vmin))
                    if metric == "delta_post_acc":
                        pad = max(0.005, pad)
                    plt.ylim(vmin - pad, vmax + pad)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        out_path = output_dir / f"{task}_{metric}.png"
        plt.savefig(out_path)
        plt.close()


def plot_average_curves(summary: pd.DataFrame, metric: str, *, amount_field: str, output_path: Path) -> None:
    if summary.empty:
        return
    keep_cols = ["strategy", "strategy_label", amount_field, f"{metric}_mean", f"{metric}_std"]
    for col in keep_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    avg = summary[keep_cols].groupby(["strategy", amount_field]).agg(
        {f"{metric}_mean": "mean", f"{metric}_std": "mean", "strategy_label": "first"}
    ).reset_index()
    plt.figure(figsize=(7, 4.5))
    for strategy, rows in avg.groupby("strategy"):
        label = rows.get("strategy_label")
        if label is not None:
            label = label.iloc[0]
        else:
            label = strategy
        plt.errorbar(
            rows[amount_field],
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"].fillna(0.0),
            fmt="-o",
            capsize=3,
            label=label,
        )
    if metric == "delta_post_acc":
        plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xlabel("prune_amount")
    plt.ylabel(f"{metric} (mean across tasks)")
    plt.title(f"Average {metric} across tasks")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, help="Suite CSV to summarise.")
    parser.add_argument("--output_prefix", default="curated", help="Prefix for summary/plot files.")
    parser.add_argument(
        "--group_fields",
        default="task,strategy,amount",
        help="Comma-separated columns to group by (default: task,strategy,amount).",
    )
    parser.add_argument(
        "--metrics",
        default="post_acc,post_loss,delta_post_acc",
        help="Comma-separated metric columns to summarise.",
    )
    parser.add_argument("--results_dir", default="results", help="Directory for summary CSV output.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to store plots.")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    metrics = tuple(col.strip() for col in args.metrics.split(",") if col.strip())
    df = _ensure_columns(df, set(metrics) | {"task", "strategy", "amount"})
    df = _normalize_columns(df)

    group_fields = tuple(col.strip() for col in args.group_fields.split(",") if col.strip())
    summary = summarise(df, group_fields=group_fields, metrics=metrics)
    if "strategy" in summary.columns:
        summary["strategy_label"] = summary["strategy"].map(lambda name: STRATEGY_LABELS.get(name, name))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{args.output_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")

    plots_dir = Path(args.plots_dir) / args.output_prefix
    amount_values = tuple(sorted(summary[group_fields[-1]].unique()))
    primary_group = group_fields[1] if len(group_fields) > 1 else "strategy"
    for metric in ("delta_post_acc", "post_acc"):
        plot_task_curves(
            summary,
            metric=metric,
            task_field=group_fields[0],
            strategy_field=primary_group,
            amount_field=group_fields[-1],
            amount_values=amount_values,
            output_dir=plots_dir,
        )
        print(f"Wrote per-task plots for {metric} to {plots_dir}")
        plot_average_curves(
            summary,
            metric=metric,
            amount_field=group_fields[-1],
            output_path=plots_dir / f"average_{metric}.png",
        )


if __name__ == "__main__":
    main()
