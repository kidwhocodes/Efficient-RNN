#!/usr/bin/env python3
"""Combine multiple pruning CSVs into a single file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine pruning results CSVs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV paths to concatenate.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Missing input CSV: {path}")

    rows = []
    fieldnames = None
    for path in input_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"CSV header mismatch: {path}")
            rows.extend(list(reader))

    if not fieldnames:
        raise SystemExit("No rows found to combine.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
