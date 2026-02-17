#!/usr/bin/env python3
"""Run a pruning sweep from a fixed baseline checkpoint with per-run progress."""

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
    parser = argparse.ArgumentParser(description="Pruning sweep from a baseline checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/scratch_prune_smoke_synthetic_seed0.pt",
        help="Baseline checkpoint path to prune.",
    )
    parser.add_argument("--task", default="synthetic")
    parser.add_argument("--decision_delay", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (overrides --seed when provided).",
    )
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=0)
    parser.add_argument("--ft_steps", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--last_only", dest="last_only", action="store_true")
    parser.add_argument("--full_sequence", dest="last_only", action="store_false")
    parser.set_defaults(last_only=True)
    parser.add_argument("--run_id", default="scratch_prune_suite")
    parser.add_argument("--output_csv", default="results/scratch_prune_suite.csv")
    parser.add_argument("--amounts", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    return parser.parse_args()


def _parse_amounts(src: str) -> List[float]:
    return [float(item.strip()) for item in src.split(",") if item.strip()]


def _parse_seeds(src: str | None, fallback: int) -> List[int]:
    if not src:
        return [fallback]
    return [int(item.strip()) for item in src.split(",") if item.strip()]


def _causal_amount_for_synapse_fraction(amount: float) -> float:
    # Keep fraction k so k^2 ~ (1 - amount) => k = sqrt(1 - amount).
    # Neuron prune amount = 1 - k.
    if amount <= 0:
        return 0.0
    if amount >= 1:
        return 1.0
    return 1.0 - (1.0 - amount) ** 0.5


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    amounts = _parse_amounts(args.amounts)
    strategies = ["random_unstructured", "l1_unstructured", "noise_prune", "causal"]
    seeds = _parse_seeds(args.seeds, args.seed)

    base_kwargs: Dict[str, object] = {
        "task": args.task,
        "device": args.device,
        "model_type": args.model_type,
        "train_steps": args.train_steps,
        "ft_steps": args.ft_steps,
        "movement_batches": args.movement_batches,
        "last_only": bool(args.last_only),
        "skip_training": True,
        "load_model_path": str(checkpoint),
    }
    if args.hidden_size is not None:
        base_kwargs["hidden_size"] = args.hidden_size
    if args.task.startswith("synthetic") and args.decision_delay is not None:
        base_kwargs["decision_delay"] = args.decision_delay

    for seed in seeds:
        base_kwargs["seed"] = seed
        for strategy in strategies:
            for amount in amounts:
                effective_amount = amount
                if strategy == "causal":
                    effective_amount = _causal_amount_for_synapse_fraction(amount)
                run_id = (
                    f"{args.run_id}_{args.task}_{strategy}_{int(amount * 100)}_seed{seed}"
                )
                print(
                    f"[run] {run_id} amount={amount} "
                    f"(effective={effective_amount:.3f})"
                )
                row = run_prune_experiment(
                    strategy=strategy,
                    amount=effective_amount,
                    prune_phase="post",
                    run_id=run_id,
                    **base_kwargs,
                )
                row["target_amount"] = amount
                append_results_csv([row], args.output_csv)
                print(
                    f"[done] {run_id} post_acc={row.get('post_acc')} "
                    f"post_loss={row.get('post_loss')}"
                )

    print(f"[suite done] results in {args.output_csv}")


if __name__ == "__main__":
    main()
