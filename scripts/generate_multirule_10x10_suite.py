#!/usr/bin/env python3
"""Generate a pruning suite config for synthetic_multirule (10 baselines x 10 trials)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a pruning suite config.")
    parser.add_argument(
        "--output",
        default="configs/prune_synthetic_multirule_10x10_suite.json",
        help="Path to write suite config JSON.",
    )
    parser.add_argument(
        "--baseline_dir",
        default="checkpoints/synthetic_multirule_10x10",
        help="Directory to store baseline checkpoints.",
    )
    parser.add_argument("--train_steps", type=int, default=2500)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--amounts",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated pruning amounts.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--trials", default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def _parse_ints(src: str) -> List[int]:
    return [int(item.strip()) for item in src.split(",") if item.strip()]


def _parse_floats(src: str) -> List[float]:
    return [float(item.strip()) for item in src.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_ints(args.seeds)
    trials = _parse_ints(args.trials)
    amounts = _parse_floats(args.amounts)
    methods = ["random_unstructured", "l1_unstructured", "noise_prune", "fisher"]

    defaults = {
        "task": "synthetic_multirule",
        "hidden_size": args.hidden_size,
        "model_type": "ctrnn",
        "device": args.device,
        "train_steps": args.train_steps,
        "ft_steps": 0,
        "last_only": True,
        "movement_batches": 20,
        "ng_T": 400,
        "ng_B": 64,
        "eval_sample_batches": 32,
        "reset_results": False,
        "resume": True,
    }

    runs = []

    # Baseline training runs.
    for seed in seeds:
        runs.append(
            {
                "run_id": f"baseline_synthetic_multirule_h{args.hidden_size}_seed{seed}",
                "strategy": "none",
                "amount": 0.0,
                "no_prune": True,
                "seed": seed,
                "save_model_path": str(
                    baseline_dir / f"baseline_synthetic_multirule_h{args.hidden_size}_seed{seed}.pt"
                ),
            }
        )

    # Pruning runs.
    for seed in seeds:
        ckpt = baseline_dir / f"baseline_synthetic_multirule_h{args.hidden_size}_seed{seed}.pt"
        for trial in trials:
            for method in methods:
                for amount in amounts:
                    run = {
                        "run_id": (
                            f"prune_synth_multirule_h{args.hidden_size}_"
                            f"{method}_{int(amount * 100)}_seed{seed}_trial{trial}"
                        ),
                        "strategy": method,
                        "amount": amount,
                        "seed": trial,
                        "skip_training": True,
                        "load_model_path": str(ckpt),
                    }
                    if method == "noise_prune":
                        run["noise_rng_seed"] = seed * 1000 + trial
                    runs.append(run)

    cfg = {
        "run_id": "prune_synthetic_multirule_10x10_suite",
        "output_csv": "results/prune_synthetic_multirule_10x10_suite.csv",
        "defaults": defaults,
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg, indent=2))
    print(f"Wrote suite config to {output_path}")


if __name__ == "__main__":
    main()
