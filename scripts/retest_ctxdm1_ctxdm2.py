#!/usr/bin/env python3
"""Train baselines for ctxdm1/ctxdm2 and report baseline + ablated accuracy."""

import argparse

import torch
import torch.nn as nn

from pruning_benchmark.experiments.runner import run_prune_experiment
from pruning_benchmark.tasks.modcog import resolve_modcog_callable, estimate_modcog_T
from pruning_benchmark.tasks.neurogym import ModCogTrialDM
from pruning_benchmark.training.loops import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--ng_T", type=int, default=0)
    parser.add_argument("--ng_B", type=int, default=64)
    return parser.parse_args()


def _build_modcog_data(task: str, args: argparse.Namespace) -> NeuroGymDatasetDM:
    env_suffix = task.split("modcog:", 1)[1].strip()
    info = resolve_modcog_callable(env_suffix)
    if info is None:
        raise ValueError(f"Unknown Mod-Cog task '{task}'")
    _, builder_fn = info
    env = builder_fn()
    T = int(args.ng_T) if args.ng_T and int(args.ng_T) > 0 else estimate_modcog_T(env)
    B = int(args.ng_B)
    return ModCogTrialDM(
        env,
        T=T,
        B=B,
        device=args.device,
        last_only=False,
        seed=args.seed,
        mask_fixation=True,
    )


def run_task(task: str, args: argparse.Namespace) -> None:
    row, model = run_prune_experiment(
        strategy="none",
        amount=0.0,
        train_steps=args.train_steps,
        ft_steps=0,
        last_only=True,
        seed=args.seed,
        device=args.device,
        task=task,
        no_prune=True,
        hidden_size=args.hidden_size,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        return_model=True,
    )
    post_acc = float(row.get("post_acc", 0.0))

    data = _build_modcog_data(task, args)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    ablated_acc = float("nan")
    with torch.no_grad():
        hidden_layer = getattr(model, "hidden_layer", None)
        if hidden_layer is not None and hasattr(hidden_layer, "weight"):
            original = hidden_layer.weight.data.clone()
            hidden_layer.weight.zero_()
            metrics = evaluate(model, data, args.device, criterion, steps=100, dataset_last_only=True)
            ablated_acc = float(metrics.get("acc", float("nan")))
            hidden_layer.weight.data.copy_(original)
    print(f"{task}\t{post_acc:.4f}\t{ablated_acc:.4f}")


def main() -> None:
    args = parse_args()
    print("task\tpost_acc\tablation_post_acc")
    for task in ("modcog:ctxdm1", "modcog:ctxdm2"):
        run_task(task, args)


if __name__ == "__main__":
    main()
