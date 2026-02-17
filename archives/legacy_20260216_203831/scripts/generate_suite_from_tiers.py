#!/usr/bin/env python3
"""
Generate a pruning suite config from the tier-classified baselines.

Example:
  python scripts/generate_suite_from_tiers.py \
      --tiers_csv results/baseline_tiers.csv \
      --manifest checkpoints/baseline_manifest.json \
      --target_tiers A \
      --model_type ctrnn \
      --tasks_per_tier 12 \
      --strategies random_unstructured,l1_unstructured,noise_prune \
      --amounts 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
      --seeds 0,1,2 \
      --output_config configs/pruning_methods_suite_tierA.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_list(src: str, cast) -> List:
    return [cast(item.strip()) for item in src.split(",") if item.strip()]


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text())
    return data


def build_run_entries(
    tasks: List[Dict],
    *,
    strategies: List[str],
    amounts: List[float],
    seeds: List[int],
) -> List[Dict]:
    runs: List[Dict] = []
    for task_info in tasks:
        task = task_info["task"]
        checkpoint = task_info["path"]
        task_slug = task.replace(":", "_")
        for seed in seeds:
            runs.append(
                {
                    "run_id": f"suite_{task_slug}_baseline_seed{seed}",
                    "strategy": "none",
                    "amount": 0.0,
                    "no_prune": True,
                    "seed": seed,
                    "task": task,
                    "load_model_path": checkpoint,
                }
            )
        for strategy in strategies:
            for amount in amounts:
                for seed in seeds:
                    entry = {
                        "run_id": f"suite_{task_slug}_{strategy}_amt{int(amount * 100)}_seed{seed}",
                        "strategy": strategy,
                        "amount": amount,
                        "seed": seed,
                        "task": task,
                        "load_model_path": checkpoint,
                    }
                    if strategy == "woodfisher":
                        entry["woodfisher_damping"] = 1e-3
                    runs.append(entry)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers_csv", required=True)
    parser.add_argument("--manifest", default="checkpoints/baseline_manifest.json")
    parser.add_argument("--target_tiers", default="A")
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--tasks_per_tier", type=int, default=12)
    parser.add_argument("--strategies", default="random_unstructured,l1_unstructured,noise_prune")
    parser.add_argument("--amounts", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--run_id", default="pruning_methods_suite_tiered")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    runid_to_path = {entry["run_id"]: path for path, entry in manifest.items()}

    target_tiers = {tier.strip() for tier in args.target_tiers.split(",") if tier.strip()}
    model_type = args.model_type.lower()

    candidates: List[Tuple[float, Dict]] = []
    with open(args.tiers_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tier") not in target_tiers:
                continue
            run_id = row["run_id"]
            checkpoint_path = runid_to_path.get(run_id)
            if not checkpoint_path:
                continue
            if f"_{model_type}_" not in run_id:
                continue
            task = row.get("task")
            if not task:
                continue
            candidates.append(
                (
                    float(row.get("pre_acc", 0.0)),
                    {
                        "task": task,
                        "path": checkpoint_path,
                    },
                )
            )

    if not candidates:
        raise SystemExit(f"No candidates found for tiers {target_tiers} and model_type {model_type}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_tasks = []
    seen_tasks = set()
    for _, info in candidates:
        if info["task"] in seen_tasks:
            continue
        selected_tasks.append(info)
        seen_tasks.add(info["task"])
        if len(selected_tasks) >= args.tasks_per_tier:
            break

    if not selected_tasks:
        raise SystemExit("No tasks selected; adjust tiers or tasks_per_tier.")

    strategies = _parse_list(args.strategies, str)
    amounts = _parse_list(args.amounts, float)
    seeds = _parse_list(args.seeds, int)

    runs = build_run_entries(selected_tasks, strategies=strategies, amounts=amounts, seeds=seeds)

    config = {
        "run_id": args.run_id,
        "output_csv": args.output_csv or f"results/{args.run_id}.csv",
        "defaults": {
            "hidden_size": 256,
            "train_steps": 0,
            "ft_steps": 0,
            "last_only": True,
            "device": "cpu",
            "movement_batches": 20,
            "ng_T": 600,
            "ng_B": 32,
            "eval_sample_batches": 32,
            "resume": True,
            "skip_training": True,
            "prune_phase": "post",
            "model_type": args.model_type.lower(),
        },
        "runs": runs,
    }

    Path(args.output_config).write_text(json.dumps(config, indent=2))
    print(f"Selected {len(selected_tasks)} tasks -> {len(runs)} runs")


if __name__ == "__main__":
    main()
