#!/usr/bin/env python3
"""Run random/L1/noise pruning trials on a Mod-Cog anti baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments.runner import append_results_csv, run_prune_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune Mod-Cog anti baseline (3 trials).")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/modcog_anti_ctrnn_seed0.pt",
        help="Baseline checkpoint path to prune.",
    )
    parser.add_argument("--task", default="modcog:anti")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--ng_T", type=int, default=600)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--last_only", dest="last_only", action="store_true")
    parser.add_argument("--full_sequence", dest="last_only", action="store_false")
    parser.set_defaults(last_only=True)
    parser.add_argument(
        "--amounts",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--run_id", default="modcog_anti_prune_trials")
    parser.add_argument(
        "--output_csv",
        default="results/modcog_anti_prune_trials.csv",
    )
    return parser.parse_args()


def _parse_amounts(src: str) -> List[float]:
    return [float(item.strip()) for item in src.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    amounts = _parse_amounts(args.amounts)
    strategies = ["random_unstructured", "l1_unstructured", "noise_prune"]

    base_kwargs: Dict[str, object] = {
        "task": args.task,
        "device": args.device,
        "model_type": args.model_type,
        "train_steps": 0,
        "ft_steps": 0,
        "movement_batches": args.movement_batches,
        "last_only": bool(args.last_only),
        "skip_training": True,
        "load_model_path": str(checkpoint),
        "ng_T": args.ng_T,
        "ng_B": args.ng_B,
        "seed": args.seed,
    }

    for trial in range(args.trials):
        trial_seed = args.seed + trial
        for strategy in strategies:
            for amount in amounts:
                run_id = (
                    f"{args.run_id}_{strategy}_{int(amount * 100)}_"
                    f"seed{args.seed}_trial{trial}"
                )
                print(f"[run] {run_id} amount={amount} trial={trial}")
                run_kwargs = dict(base_kwargs)
                run_kwargs["seed"] = trial_seed
                if strategy == "noise_prune":
                    run_kwargs["noise_rng_seed"] = trial_seed
                row = run_prune_experiment(
                    strategy=strategy,
                    amount=amount,
                    prune_phase="post",
                    run_id=run_id,
                    **run_kwargs,
                )
                append_results_csv([row], args.output_csv)
                print(f"[done] {run_id} post_acc={row.get('post_acc')}")

    print(f"[suite done] results in {args.output_csv}")


if __name__ == "__main__":
    main()
