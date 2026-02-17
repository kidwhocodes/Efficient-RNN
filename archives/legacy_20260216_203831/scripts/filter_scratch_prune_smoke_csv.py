#!/usr/bin/env python3
"""Filter scratch_prune_smoke.csv to a single run_id prefix + task."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter scratch_prune_smoke.csv.")
    parser.add_argument(
        "--input_csv",
        default="results/scratch_prune_smoke.csv",
        help="Source CSV to filter.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/scratch_prune_smoke_filtered.csv",
        help="Filtered CSV output path.",
    )
    parser.add_argument(
        "--run_prefix",
        default="scratch_prune_smoke_synthetic_seed0",
        help="Only keep rows whose run_id starts with this prefix.",
    )
    parser.add_argument(
        "--task",
        default="synthetic",
        help="Only keep rows matching this task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows: List[Dict[str, str]] = []
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", "")
            task = row.get("task", "")
            if run_id.startswith(args.run_prefix) and task == args.task:
                rows.append(row)

    if not rows:
        raise SystemExit("No rows matched the filter. Check --run_prefix and --task.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
