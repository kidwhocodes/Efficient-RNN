#!/usr/bin/env python3
"""Compare noise-prune hyperparameters vs random/L1 on synthetic_multirule."""

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
    parser = argparse.ArgumentParser(description="Noise-prune hyperparameter sweep.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/baseline_synthetic_multirule_h128_seed0.pt",
        help="Baseline checkpoint path to prune.",
    )
    parser.add_argument("--task", default="synthetic_multirule")
    parser.add_argument("--decision_delay", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--last_only", dest="last_only", action="store_true")
    parser.add_argument("--full_sequence", dest="last_only", action="store_false")
    parser.set_defaults(last_only=True)
    parser.add_argument("--run_id", default="noise_prune_hparam_sweep")
    parser.add_argument("--output_csv", default="results/noise_prune_hparam_sweep.csv")
    parser.add_argument("--amounts", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    return parser.parse_args()


def _parse_amounts(src: str) -> List[float]:
    return [float(item.strip()) for item in src.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    amounts = _parse_amounts(args.amounts)
    strategies = ["random_unstructured", "l1_unstructured"]
    noise_variants: List[Dict[str, object]] = [
        {
            "label": "sigma1_eps0.3_md1",
            "noise_sigma": 1.0,
            "noise_eps": 0.3,
            "noise_matched_diagonal": True,
        },
        {
            "label": "sigma0.5_eps0.3_md1",
            "noise_sigma": 0.5,
            "noise_eps": 0.3,
            "noise_matched_diagonal": True,
        },
        {
            "label": "sigma2_eps0.3_md1",
            "noise_sigma": 2.0,
            "noise_eps": 0.3,
            "noise_matched_diagonal": True,
        },
        {
            "label": "sigma1_eps0.2_md1",
            "noise_sigma": 1.0,
            "noise_eps": 0.2,
            "noise_matched_diagonal": True,
        },
        {
            "label": "sigma1_eps0.3_md0",
            "noise_sigma": 1.0,
            "noise_eps": 0.3,
            "noise_matched_diagonal": False,
        },
    ]

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
        "seed": args.seed,
    }
    if args.task.startswith("synthetic") and args.decision_delay is not None:
        base_kwargs["decision_delay"] = args.decision_delay

    for amount in amounts:
        for strategy in strategies:
            run_id = f"{args.run_id}_{args.task}_{strategy}_{int(amount * 100)}_seed{args.seed}"
            print(f"[run] {run_id} amount={amount}")
            row = run_prune_experiment(
                strategy=strategy,
                amount=amount,
                prune_phase="post",
                run_id=run_id,
                **base_kwargs,
            )
            append_results_csv([row], args.output_csv)
            print(f"[done] {run_id} post_acc={row.get('post_acc')}")

        for variant in noise_variants:
            label = variant["label"]
            prune_kwargs = {k: v for k, v in variant.items() if k != "label"}
            run_id = (
                f"{args.run_id}_{args.task}_noise_{label}_"
                f"{int(amount * 100)}_seed{args.seed}"
            )
            print(f"[run] {run_id} amount={amount}")
            row = run_prune_experiment(
                strategy="noise_prune",
                amount=amount,
                prune_phase="post",
                run_id=run_id,
                **base_kwargs,
                **prune_kwargs,
            )
            append_results_csv([row], args.output_csv)
            print(f"[done] {run_id} post_acc={row.get('post_acc')}")

    print(f"[suite done] results in {args.output_csv}")


if __name__ == "__main__":
    main()
