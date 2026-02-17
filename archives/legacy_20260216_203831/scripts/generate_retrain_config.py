#!/usr/bin/env python3
"""
Generate a baseline config for retraining low-tier Mod-Cog tasks.

Reads results/baseline_tiers.csv (from classify_modcog_baselines.py) and builds
configs/baselines_modcog_retrain_low.json containing the targeted tasks with
extended train_steps/ft_steps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


def load_target_runs(tiers_csv: Path, tiers: set[str]) -> Dict[Tuple[str, str], Dict]:
    targets: Dict[Tuple[str, str], Dict] = {}
    with open(tiers_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task", "")
            tier = row.get("tier", "")
            if tier not in tiers or not task.startswith("modcog:"):
                continue
            run_id = row["run_id"]
            targets[(task, run_id)] = row
    return targets


def build_task_entry(run_id: str, config_path: Path, train_steps: int, ft_steps: int) -> Dict:
    config = json.loads(config_path.read_text())
    experiment = config.get("experiment", {})
    task_meta = config.get("task", {})
    model_meta = config.get("model", {})
    task = experiment.get("task", task_meta.get("task"))
    if not task:
        raise ValueError(f"Unable to determine task for {run_id}")
    model_type = str(experiment.get("model_type", model_meta.get("class", "ctrnn"))).lower()
    save_path = experiment.get("save_model_path", f"{task.replace(':', '_')}_{model_type}_seed{experiment.get('seed', 0)}.pt")
    checkpoint_name = Path(save_path).name
    run_prefix = f"retrain_{Path(checkpoint_name).stem}"
    entry = {
        "task": task,
        "seeds": [int(experiment.get("seed", 0))],
        "checkpoint_name": checkpoint_name,
        "run_id_prefix": run_prefix,
        "model_type": model_type,
        "hidden_size": int(model_meta.get("hidden_size", experiment.get("hidden_size", 256))),
        "ng_T": task_meta.get("T", experiment.get("ng_T", 600)),
        "ng_B": task_meta.get("B", experiment.get("ng_B", 32)),
        "train_steps": train_steps,
        "ft_steps": ft_steps,
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers_csv", default="results/baseline_tiers.csv")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_config", default="configs/baselines_modcog_retrain_low.json")
    parser.add_argument("--target_tiers", default="C", help="Comma-separated list of tiers to retrain (default: C).")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--ft_steps", type=int, default=100)
    args = parser.parse_args()

    tiers = set(t.strip() for t in args.target_tiers.split(",") if t.strip())
    targets = load_target_runs(Path(args.tiers_csv), tiers)
    if not targets:
        raise SystemExit(f"No runs matched tiers {tiers} in {args.tiers_csv}.")

    task_entries: Dict[Tuple[str, str], Dict] = {}
    for (_, run_id), row in targets.items():
        config_path = Path(args.results_dir) / run_id / "config.json"
        if not config_path.exists():
            print(f"[warn] config not found for {run_id}, skipping")
            continue
        task = row["task"]
        # key by task + model to avoid duplicates
        config = json.loads(config_path.read_text())
        model_type = str(config.get("experiment", {}).get("model_type", config.get("model", {}).get("class", "ctrnn"))).lower()
        key = (task, model_type)
        if key in task_entries:
            continue
        entry = build_task_entry(run_id, config_path, args.train_steps, args.ft_steps)
        task_entries[key] = entry

    tasks = list(task_entries.values())
    if not tasks:
        raise SystemExit("No task entries generated; check the tiers or results directory.")

    config = {
        "run_id": "retrain_modcog_low_tier",
        "output_dir": "checkpoints",
        "defaults": {
            "hidden_size": 256,
            "train_steps": args.train_steps,
            "ft_steps": args.ft_steps,
            "last_only": True,
            "device": "cpu",
            "movement_batches": 20,
            "ng_T": 600,
            "ng_B": 32,
            "eval_sample_batches": 32,
            "resume": False,
        },
        "tasks": tasks,
    }
    Path(args.output_config).write_text(json.dumps(config, indent=2))
    print(f"Wrote retrain config with {len(tasks)} tasks -> {args.output_config}")


if __name__ == "__main__":
    main()
