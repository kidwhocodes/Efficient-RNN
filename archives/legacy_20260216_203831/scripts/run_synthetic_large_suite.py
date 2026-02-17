#!/usr/bin/env python3
"""Run a large synthetic suite (3 tasks x 3 baselines x 10 trials per method)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments.baseline import train_baselines
from pruning_benchmark.experiments.harness import run_suite_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a large synthetic pruning suite.")
    parser.add_argument(
        "--tasks",
        default="synthetic,synthetic_context,synthetic_multirule",
        help="Comma-separated synthetic tasks to include.",
    )
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=2500)
    parser.add_argument("--decision_delay", type=int, default=20)
    parser.add_argument(
        "--strategies",
        default="random_unstructured,l1_unstructured,noise_prune,fisher,movement",
        help="Comma-separated pruning strategies.",
    )
    parser.add_argument("--amounts", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--baseline_seeds", default="0,1,2")
    parser.add_argument("--trial_seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--output_csv", default="results/synthetic_large_suite.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--eval_sample_batches", type=int, default=32)
    return parser.parse_args()


def _parse_list(src: str, cast):
    items = [item.strip() for item in src.split(",") if item.strip()]
    return [cast(item) for item in items]


def main() -> None:
    args = parse_args()
    tasks = _parse_list(args.tasks, str)
    strategies = _parse_list(args.strategies, str)
    amounts = _parse_list(args.amounts, float)
    baseline_seeds = _parse_list(args.baseline_seeds, int)
    trial_seeds = _parse_list(args.trial_seeds, int)

    config_dir = ROOT / "configs"
    config_dir.mkdir(exist_ok=True)

    baseline_cfg = {
        "output_dir": "checkpoints",
        "defaults": {
            "hidden_size": args.hidden_size,
            "train_steps": args.train_steps,
            "ft_steps": 0,
            "last_only": True,
            "device": args.device,
            "movement_batches": args.movement_batches,
            "ng_T": 0,
            "ng_B": 0,
            "decision_delay": args.decision_delay,
        },
        "tasks": [
            {
                "task": task,
                "seeds": baseline_seeds,
                "run_id_prefix": f"baseline_{task}_h{args.hidden_size}",
                "checkpoint_name": f"baseline_{task}_h{args.hidden_size}_seed{{seed}}.pt",
            }
            for task in tasks
        ],
    }
    baseline_cfg_path = config_dir / "baseline_synthetic_large_suite.json"
    baseline_cfg_path.write_text(json.dumps(baseline_cfg, indent=2))
    print(f"[baseline] config -> {baseline_cfg_path}")

    prune_runs = []
    for task in tasks:
        for base_seed in baseline_seeds:
            ckpt = f"checkpoints/baseline_{task}_h{args.hidden_size}_seed{base_seed}.pt"
            for trial_seed in trial_seeds:
                for strat in strategies:
                    for amt in amounts:
                        prune_runs.append(
                            {
                                "run_id": (
                                    f"prune_{task}_h{args.hidden_size}_base{base_seed}_"
                                    f"{strat}_{int(amt * 100)}_seed{trial_seed}"
                                ),
                                "strategy": strat,
                                "amount": amt,
                                "seed": trial_seed,
                                "skip_training": True,
                                "load_model_path": ckpt,
                                "prune_phase": "post",
                                "task": task,
                            }
                        )

    prune_cfg = {
        "run_id": "synthetic_large_suite",
        "output_csv": args.output_csv,
        "defaults": {
            "hidden_size": args.hidden_size,
            "train_steps": 0,
            "ft_steps": 0,
            "last_only": True,
            "device": args.device,
            "movement_batches": args.movement_batches,
            "ng_T": 0,
            "ng_B": 0,
            "decision_delay": args.decision_delay,
            "eval_sample_batches": args.eval_sample_batches,
            "reset_results": True,
        },
        "runs": prune_runs,
    }
    prune_cfg_path = config_dir / "prune_synthetic_large_suite.json"
    prune_cfg_path.write_text(json.dumps(prune_cfg, indent=2))
    print(f"[prune] config -> {prune_cfg_path}")

    print("[baseline] training/reusing checkpoints...")
    train_baselines(str(baseline_cfg_path), overwrite=False)

    print("[suite] running pruning suite...")
    run_suite_from_config(str(prune_cfg_path))

    print(f"[done] results -> {args.output_csv}")


if __name__ == "__main__":
    main()
