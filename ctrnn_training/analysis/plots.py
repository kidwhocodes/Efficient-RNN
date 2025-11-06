"""Plotting helpers for summarising pruning experiments."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .summary import summarize_csv


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def plot_metrics(
    csv_path: str,
    metrics: Sequence[str],
    group_field: str = "strategy",
    output_dir: str = "plots",
    amount_field: str = "amount",
) -> List[str]:
    """
    Generate line plots for each metric vs. amount grouped by `group_field`.

    Returns the list of file paths written.
    """
    summaries = summarize_csv(
        csv_path,
        group_fields=(group_field, amount_field),
        metrics=metrics,
        output_path=None,
    )
    if not summaries:
        raise ValueError("No data available to plot. Check the CSV or grouping fields.")

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    # Organise data: {group -> {metric -> list of (amount, mean, std, count)}}
    grouped: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in summaries:
        group = row[group_field]
        amt = _to_float(row[amount_field])
        for metric in metrics:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            count = row.get(f"{metric}_count")
            if mean is None:
                continue
            grouped[group][metric].append(
                {
                    "amount": amt,
                    "mean": float(mean),
                    "std": float(std) if std is not None else 0.0,
                    "count": int(count) if count is not None else 1,
                }
            )

    for metric in metrics:
        plt.figure(figsize=(7, 4.5))
        for group, metric_data in grouped.items():
            points = metric_data.get(metric)
            if not points:
                continue
            points = [p for p in points if not (np.isnan(p["mean"]) or np.isinf(p["mean"]))]
            if not points:
                continue
            points.sort(key=lambda p: p["amount"])
            xs = [p["amount"] for p in points]
            ys = [p["mean"] for p in points]
            errs = [p["std"] for p in points]
            fmt = "-o" if len(xs) > 1 else "o"
            plt.errorbar(
                xs,
                ys,
                yerr=errs,
                fmt=fmt,
                capsize=3,
                label=group,
            )
        plt.xlabel(amount_field)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {amount_field}")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        out_path = os.path.join(output_dir, f"{metric}_by_{group_field}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        written.append(out_path)
    return written


__all__ = ["plot_metrics"]
