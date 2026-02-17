#!/usr/bin/env python3
"""
Quick inspection utility for pruning CSVs.

It prints, per strategy, the unique values observed for the `prune_feedforward`
flag along with summary statistics for the input/readout sparsity columns so
you can confirm whether the feed-forward layers were actually pruned.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect feed-forward pruning metadata.")
    parser.add_argument("--csv", required=True, help="Path to a pruning results CSV.")
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task filter (e.g., 'modcog:ctxdlydm1seql').",
    )
    return parser.parse_args()


def summarize(rows, task_filter: str | None):
    required = {
        "task",
        "strategy",
        "prune_feedforward",
        "post_sparsity_input",
        "post_sparsity_readout",
    }
    if not rows:
        raise ValueError("CSV is empty.")
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    stats: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "input_sum": 0.0,
            "input_min": float("inf"),
            "input_max": float("-inf"),
            "readout_sum": 0.0,
            "readout_min": float("inf"),
            "readout_max": float("-inf"),
        }
    )

    for row in rows:
        if task_filter and row["task"] != task_filter:
            continue
        key = (row["task"], row["strategy"], row.get("prune_feedforward", ""))
        try:
            inp = float(row["post_sparsity_input"])
            out = float(row["post_sparsity_readout"])
        except ValueError:
            continue
        entry = stats[key]
        entry["count"] += 1
        entry["input_sum"] += inp
        entry["readout_sum"] += out
        entry["input_min"] = min(entry["input_min"], inp)
        entry["input_max"] = max(entry["input_max"], inp)
        entry["readout_min"] = min(entry["readout_min"], out)
        entry["readout_max"] = max(entry["readout_max"], out)

    return stats


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    stats = summarize(rows, args.task)
    if not stats:
        raise ValueError("No rows matched the requested filter.")
    print(f"{'task':30} {'strategy':20} {'feedforward':13} {'in_mean':8} {'in_min':8} {'in_max':8} {'out_mean':9} {'out_min':8} {'out_max':8}")
    for (task, strategy, feedforward), entry in sorted(stats.items()):
        count = entry["count"]
        in_mean = entry["input_sum"] / max(1, count)
        out_mean = entry["readout_sum"] / max(1, count)
        print(
            f"{task:30} {strategy:20} {feedforward:13} "
            f"{in_mean:8.3f} {entry['input_min']:8.3f} {entry['input_max']:8.3f} "
            f"{out_mean:9.3f} {entry['readout_min']:8.3f} {entry['readout_max']:8.3f}"
        )


if __name__ == "__main__":
    main()
