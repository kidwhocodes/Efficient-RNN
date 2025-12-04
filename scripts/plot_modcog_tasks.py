#!/usr/bin/env python3
"""Generate per-task pruning plots from a Mod_Cog results CSV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Sequence

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
