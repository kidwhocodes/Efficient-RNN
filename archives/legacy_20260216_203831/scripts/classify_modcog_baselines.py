#!/usr/bin/env python3
"""
Classify baseline checkpoints into tiers based on their pre-accuracy.

Scans results/baseline_* directories, extracts task/model/pre_acc, and writes
results/baseline_tiers.csv + .json so suites can filter by high/low performers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_baseline_rows(results_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for run_dir in results_dir.glob("baseline_*"):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        config = json.loads(config_path.read_text())
        task = config.get("task", {}).get("task", "unknown")
        model = config.get("model", {}).get("class", config.get("experiment", {}).get("model_type", "unknown"))
        model = str(model).lower()
        pre_acc = metrics.get("pre_acc")
        if pre_acc is None:
            pre_acc = metrics.get("post_acc", 0.0)
        rows.append({
            "run_id": run_dir.name,
            "task": task,
            "model": model,
            "pre_acc": float(pre_acc),
            "metrics_path": str(metrics_path),
        })
    return rows


def assign_tiers(rows: List[Dict], *, tier_a: float, tier_b: float) -> List[Dict]:
    tiered = []
    for row in rows:
        acc = row["pre_acc"]
        if acc >= tier_a:
            tier = "A"
        elif acc >= tier_b:
            tier = "B"
        else:
            tier = "C"
        tiered.append({**row, "tier": tier})
    return tiered


def write_outputs(rows: List[Dict], output_prefix: Path) -> None:
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    fieldnames = ["run_id", "task", "model", "pre_acc", "tier", "metrics_path"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} rows to {csv_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results", help="Directory containing baseline_* runs.")
    parser.add_argument("--output_prefix", default="results/baseline_tiers", help="Prefix for the CSV/JSON outputs.")
    parser.add_argument("--tier_a", type=float, default=0.8, help="pre_acc threshold for Tier A.")
    parser.add_argument("--tier_b", type=float, default=0.2, help="pre_acc threshold for Tier B (below is Tier C).")
    args = parser.parse_args()

    rows = load_baseline_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit("No baseline runs found under results/. Run the baseline training first.")
    rows = assign_tiers(rows, tier_a=args.tier_a, tier_b=args.tier_b)
    write_outputs(rows, Path(args.output_prefix))


if __name__ == "__main__":
    main()
