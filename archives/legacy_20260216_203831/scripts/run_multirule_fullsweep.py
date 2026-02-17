#!/usr/bin/env python3
"""Generate and execute the full multirule pruning sweep."""

from __future__ import annotations

import json
from pathlib import Path

from pruning_benchmark.experiments import run_suite_from_config

CONFIG_PATH = Path("configs/multirule_fullsweep.json")
OUTPUT_CSV = "results/multirule_fullsweep.csv"

STRATEGIES = [
    "random_unstructured",
    "l1_unstructured",
    "movement",
    "snip",
    "synflow",
    "fisher",
    "grasp",
    "obd",
    "set",
    "noise_prune",
]

AMOUNTS = [round(x / 10.0, 1) for x in range(1, 10)]
SEEDS = [0, 1, 2]


def build_config() -> dict:
    defaults = {
        "task": "synthetic_multirule",
        "hidden_size": 256,
        "train_steps": 500,
        "ft_steps": 10,
        "last_only": True,
        "device": "cpu",
        "movement_batches": 20,
        "reset_results": True,
    }
    runs: list[dict] = []
    for seed in SEEDS:
        checkpoint = f"checkpoints/multirule_fullsweep_seed{seed}.pt"
        runs.append(
            {
                "run_id": f"multirule_full_baseline_seed{seed}",
                "strategy": "none",
                "amount": 0.0,
                "no_prune": True,
                "seed": seed,
                "train_steps": 500,
                "save_model_path": checkpoint,
            }
        )
        for strategy in STRATEGIES:
            for amount in AMOUNTS:
                suffix = f"{strategy}_{int(amount * 100)}_seed{seed}"
                runs.append(
                    {
                        "run_id": f"multirule_full_{suffix}",
                        "strategy": strategy,
                        "amount": amount,
                        "seed": seed,
                        "skip_training": True,
                        "load_model_path": checkpoint,
                    }
                )
    return {
        "run_id": "multirule_fullsweep",
        "output_csv": OUTPUT_CSV,
        "defaults": defaults,
        "runs": runs,
    }


def ensure_config_file() -> Path:
    config = build_config()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    return CONFIG_PATH


def main() -> None:
    config_path = ensure_config_file()
    csv_path = run_suite_from_config(str(config_path))
    print(f"[done] Suite results written to {csv_path}")
    print(f"[info] Config saved to {config_path}")


if __name__ == "__main__":
    main()
