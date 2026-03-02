#!/usr/bin/env python3
"""Compute overlap of nonzero recurrent weights between pruning strategies."""

import argparse
from itertools import combinations

import torch

from pruning_benchmark.experiments.runner import run_prune_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to baseline checkpoint to prune",
    )
    parser.add_argument("--amount", type=float, required=True, help="Pruning amount (e.g. 0.5)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for task sampling and RNG")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default="modcog:ctxdm2")
    parser.add_argument("--noise_rng_seed", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=1)
    return parser.parse_args()


def _nonzero_mask(model) -> torch.Tensor:
    weight = model.hidden_layer.weight.detach().cpu()
    return weight != 0


def main() -> None:
    args = parse_args()
    strategies = ["noise_prune", "l1_unstructured", "random_unstructured", "obd"]

    masks = {}
    counts = {}

    for strategy in strategies:
        extra = {}
        if strategy == "noise_prune" and args.noise_rng_seed is not None:
            extra["noise_rng_seed"] = args.noise_rng_seed
        row, model = run_prune_experiment(
            strategy=strategy,
            amount=args.amount,
            train_steps=0,
            ft_steps=0,
            last_only=True,
            seed=args.seed,
            device=args.device,
            task=args.task,
            skip_training=True,
            load_model_path=args.checkpoint,
            eval_steps_pre0=args.eval_steps,
            eval_steps_pre=args.eval_steps,
            eval_steps_post0=args.eval_steps,
            eval_steps_post=args.eval_steps,
            return_model=True,
            **extra,
        )
        mask = _nonzero_mask(model)
        masks[strategy] = mask
        counts[strategy] = int(mask.sum().item())

    print("strategy\tnonzero_count")
    for strategy in strategies:
        print(f"{strategy}\t{counts[strategy]}")

    print("\nstrategy_a\tstrategy_b\toverlap_count\toverlap_pct_a\toverlap_pct_b\toverlap_pct_union")
    for a, b in combinations(strategies, 2):
        overlap = int((masks[a] & masks[b]).sum().item())
        count_a = counts[a]
        count_b = counts[b]
        union = int((masks[a] | masks[b]).sum().item())
        pct_a = 100.0 * overlap / count_a if count_a else 0.0
        pct_b = 100.0 * overlap / count_b if count_b else 0.0
        pct_union = 100.0 * overlap / union if union else 0.0
        print(
            f"{a}\t{b}\t{overlap}\t"
            f"{pct_a:.2f}\t{pct_b:.2f}\t{pct_union:.2f}"
        )


if __name__ == "__main__":
    main()
