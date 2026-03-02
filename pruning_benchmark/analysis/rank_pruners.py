"""Rank pruning methods by average metric."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Rank pruning methods.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--metric", default="post_acc")
    parser.add_argument("--group_field", default="strategy")
    parser.add_argument("--filter_task", default=None)
    args = parser.parse_args()

    totals = defaultdict(float)
    counts = defaultdict(int)

    with open(args.input_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get(args.group_field) == "none":
                continue
            if args.filter_task and row.get("task") != args.filter_task:
                continue
            try:
                val = float(row.get(args.metric, "nan"))
            except (TypeError, ValueError):
                continue
            if val != val:
                continue
            group = row.get(args.group_field)
            totals[group] += val
            counts[group] += 1

    scores = [(grp, totals[grp] / counts[grp]) for grp in totals if counts[grp] > 0]
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Ranking ({args.metric}):")
    for rank, (grp, score) in enumerate(scores, 1):
        print(f"{rank:>2}. {grp:20s} {score:.4f} (n={counts[grp]})")


if __name__ == "__main__":
    main()
