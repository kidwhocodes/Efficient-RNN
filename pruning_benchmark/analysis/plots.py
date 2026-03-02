"""Plotting helpers for summarising pruning experiments."""

from __future__ import annotations

import csv
import os
import tempfile
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
    filters: Dict[str, str] | None = None,
    show_error_bars: bool = True,
) -> List[str]:
    """
    Generate line plots for each metric vs. amount grouped by `group_field`.

    Returns the list of file paths written.
    """
    amount_field_to_use = amount_field
    task_label = ""
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        tasks = {row.get("task") for row in reader if row.get("task")}
    if not tasks:
        task_label = "Task"
    elif len(tasks) == 1:
        task_label = next(iter(tasks))
    else:
        task_label = f"Task Suite ({len(tasks)} tasks)"
    temp_csv_path = None
    if amount_field == "amount":
        with open(csv_path, newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if "post_sparsity_recurrent" in header:
            amount_field_to_use = "amount_pruned"
            with open(csv_path, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if not rows:
                raise ValueError("No data available to plot. Check the CSV or grouping fields.")
            fieldnames = list(rows[0].keys())
            if "amount_pruned" not in fieldnames:
                fieldnames.append("amount_pruned")
            step = 0.05
            for row in rows:
                raw = row.get("post_sparsity_recurrent", "")
                try:
                    value = float(raw)
                    quantized = round(round(value / step) * step, 2)
                    row["amount_pruned"] = f"{quantized:.2f}"
                except (TypeError, ValueError):
                    row["amount_pruned"] = row.get("amount", "")
            tmp = tempfile.NamedTemporaryFile("w", newline="", delete=False)
            with tmp:
                writer = csv.DictWriter(tmp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            temp_csv_path = tmp.name

    summaries = summarize_csv(
        temp_csv_path or csv_path,
        group_fields=(group_field, amount_field_to_use),
        metrics=metrics,
        output_path=None,
        filters=filters,
    )
    if not summaries:
        raise ValueError("No data available to plot. Check the CSV or grouping fields.")

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    # Organise data: {group -> {metric -> list of (amount, mean, std, count)}}
    grouped: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in summaries:
        group = row[group_field]
        amt = _to_float(row[amount_field_to_use])
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
            if show_error_bars:
                plt.errorbar(
                    xs,
                    ys,
                    yerr=errs,
                    fmt=fmt,
                    capsize=3,
                    label=group,
                )
            else:
                plt.plot(xs, ys, fmt, label=group)
        if "acc" in metric:
            plt.xlabel("Amount Pruned")
            plt.ylabel("Task Accuracy")
            plt.title(f"{task_label}: Task Accuracy vs Amount Pruned")
        else:
            plt.xlabel("Amount Pruned")
            plt.ylabel(metric)
            plt.title(f"{task_label}: {metric} vs Amount Pruned")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        out_path = os.path.join(output_dir, f"{metric}_by_{group_field}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        written.append(out_path)
    return written


__all__ = ["plot_metrics"]
