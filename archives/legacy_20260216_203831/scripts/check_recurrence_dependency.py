#!/usr/bin/env python3
"""Sanity check that task accuracy depends on recurrent dynamics."""

from __future__ import annotations

import argparse

import torch

from pruning_benchmark.experiments import run_prune_experiment
from pruning_benchmark.tasks.synthetic import SynthCfg, SyntheticDM
from pruning_benchmark.training import evaluate


def _zero_recurrent(model: torch.nn.Module) -> None:
    if not hasattr(model, "hidden_layer"):
        raise ValueError("Model has no recurrent hidden_layer to ablate.")
    with torch.no_grad():
        model.hidden_layer.weight.zero_()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check recurrence dependence on synthetic task.")
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--decision_delay", type=int, default=5)
    parser.add_argument("--min_drop", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    row, model = run_prune_experiment(
        strategy="none",
        amount=0.0,
        train_steps=args.train_steps,
        ft_steps=0,
        last_only=True,
        seed=0,
        device=args.device,
        movement_batches=1,
        task="synthetic",
        no_prune=True,
        decision_delay=args.decision_delay,
        return_model=True,
        run_id="recurrence_sanity",
    )

    cfg = SynthCfg(
        T=60,
        B=64,
        coh_levels=(0.0, 0.05, 0.1, 0.2),
        stim_std=0.6,
        decision_delay=args.decision_delay,
    )
    data = SyntheticDM(cfg)
    criterion = torch.nn.CrossEntropyLoss()

    baseline = evaluate(
        model,
        data,
        args.device,
        criterion,
        steps=args.eval_steps,
        dataset_last_only=True,
        eval_last_only=True,
    )

    _zero_recurrent(model)
    ablated = evaluate(
        model,
        data,
        args.device,
        criterion,
        steps=args.eval_steps,
        dataset_last_only=True,
        eval_last_only=True,
    )

    drop = baseline["acc"] - ablated["acc"]
    print(f"baseline_acc={baseline['acc']:.3f} ablated_acc={ablated['acc']:.3f} drop={drop:.3f}")
    if drop < args.min_drop:
        raise SystemExit(
            f"Recurrent ablation drop {drop:.3f} below threshold {args.min_drop:.3f}."
        )

    print("Recurrence dependence check passed.")


if __name__ == "__main__":
    main()
