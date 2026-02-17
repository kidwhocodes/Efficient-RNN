#!/usr/bin/env python3
"""Generate per-task or aggregated pruning plots from a Mod_Cog results CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

# ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.analysis.plots import plot_metrics


def _parse_list(src: str | None) -> List[str]:
    if not src:
        return []
    return [item.strip() for item in src.split(",") if item.strip()]


def _sanitize_dir_name(task: str) -> str:
    slug = task.replace("modcog:", "")
    slug = slug.replace("Mod_Cog-", "")
    slug = slug.replace(":", "_").replace("/", "_")
    return slug


def _discover_tasks(csv_path: str) -> List[str]:
    tasks: List[str] = []
    seen = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task")
            if not task or task in seen:
                continue
            seen.add(task)
            tasks.append(task)
    return tasks


def plot_per_task(
    csv_path: str,
    tasks: Sequence[str],
    metrics: Sequence[str],
    output_root: str,
    group_field: str = "strategy",
    amount_field: str = "amount",
) -> List[str]:
    written: List[str] = []
    for task in tasks:
        slug = _sanitize_dir_name(task)
        out_dir = os.path.join(output_root, slug)
        os.makedirs(out_dir, exist_ok=True)
        filters = {"task": task}
        print(f"[info] Plotting {task} -> {out_dir}")
        files = plot_metrics(
            csv_path,
            metrics=metrics,
            group_field=group_field,
            amount_field=amount_field,
            output_dir=out_dir,
            filters=filters,
        )
        written.extend(files)
    return written


def plot_aggregated(
    csv_path: str,
    metrics: Sequence[str],
    output_dir: str,
    *,
    group_field: str = "strategy",
    amount_field: str = "amount",
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    def _to_float(val: str) -> float | None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    for metric in metrics:
        agg: dict[str, dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            group = row.get(group_field)
            amount = _to_float(row.get(amount_field))
            value = _to_float(row.get(metric))
            if group is None or amount is None or value is None:
                continue
            agg[group][amount].append(value)

        if not agg:
            continue

        import matplotlib.pyplot as plt  # imported lazily to avoid backend issues

        plt.figure(figsize=(8, 5))
        for group in sorted(agg):
            pts = sorted(agg[group].items())
            xs = [amt for amt, _ in pts]
            ys = [sum(vals) / len(vals) for _, vals in pts]
            plt.plot(xs, ys, marker="o", label=group)
        plt.xlabel(amount_field)
        plt.ylabel(f"{metric} (mean)")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{metric}_mean_by_{group_field}.png")
        plt.savefig(out_path)
        plt.close()
        written.append(out_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Mod_Cog pruning results per task.")
    parser.add_argument("--input_csv", required=True, help="CSV file produced by a pruning suite.")
    parser.add_argument(
        "--output_root",
        default="plots/modcog_tasks",
        help="Directory where per-task subfolders will be created.",
    )
    parser.add_argument(
        "--metrics",
        default="post_acc,post_loss",
        help="Comma-separated metrics to plot (must match CSV columns, e.g., post_acc).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated list of task IDs (e.g., modcog:Go-v0). Defaults to all tasks in the CSV.",
    )
    parser.add_argument(
        "--group_field",
        default="strategy",
        help="CSV column used for grouping lines (default: strategy).",
    )
    parser.add_argument(
        "--amount_field",
        default="amount",
        help="CSV column representing pruning amounts (default: amount).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="If set, aggregate across all tasks and plot the mean metric per strategy/amount.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.input_csv
    metrics = _parse_list(args.metrics)
    if not metrics:
        raise SystemExit("No metrics provided.")
    tasks = _parse_list(args.tasks)
    if not tasks:
        tasks = _discover_tasks(csv_path)
        if not tasks:
            raise SystemExit("No tasks found in the CSV.")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if args.combined:
        written = plot_aggregated(
            csv_path,
            metrics=metrics,
            output_dir=args.output_root,
            group_field=args.group_field,
            amount_field=args.amount_field,
        )
    else:
        written = plot_per_task(
            csv_path,
            tasks=tasks,
            metrics=metrics,
            output_root=args.output_root,
            group_field=args.group_field,
            amount_field=args.amount_field,
        )
    print(f"[done] Wrote {len(written)} plot files under {args.output_root}")


if __name__ == "__main__":
    main()
