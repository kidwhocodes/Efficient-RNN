#!/usr/bin/env python3
"""Train a Mod-Cog baseline and report ablated accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.register_modcog_envs import register_envs

from pruning_benchmark.experiments.runner import run_prune_experiment
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import NeuroGymDatasetDM
from pruning_benchmark.training import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mod-Cog baseline and ablate recurrence.")
    parser.add_argument("--task", default="modcog:ctxdlydm2intseq")
    parser.add_argument("--train_steps", type=int, default=2500)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ng_T", type=int, default=600)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--mask_last_k", type=int, default=0)
    parser.add_argument("--mask_context_after_t", type=int, default=None)
    parser.add_argument(
        "--context_indices",
        type=str,
        default="",
        help="Comma-separated indices to mask after mask_context_after_t.",
    )
    parser.add_argument(
        "--context_top_k",
        type=int,
        default=2,
        help="Auto-select top-k context indices when none are provided.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/modcog_ctxdlydm2intseq_h128_seed0.pt",
        help="Checkpoint output path.",
    )
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--run_id", default="baseline_modcog_ctxdlydm2intseq_h128_seed0")
    parser.add_argument(
        "--full_sequence",
        action="store_true",
        help="Evaluate accuracy over the full sequence instead of last step only.",
    )
    parser.add_argument(
        "--strict_ablation",
        action="store_true",
        help="Disable recurrent weights and state carryover (feedforward-only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_envs()

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    eval_last_only = not args.full_sequence
    context_indices = []
    if args.context_indices:
        context_indices = [int(x.strip()) for x in args.context_indices.split(",") if x.strip()]
    elif args.mask_context_after_t is not None and args.context_top_k > 0:
        env_id = ensure_modcog_env_id(args.task)
        if env_id is None:
            raise ValueError(f"Unknown Mod_Cog task: {args.task}")
        probe_data = NeuroGymDatasetDM(
            env_id,
            T=args.ng_T,
            B=args.ng_B,
            device=args.device,
            last_only=False,
            seed=args.seed,
        )
        x_batch, _ = probe_data.sample_batch()
        cutoff = min(args.mask_context_after_t, x_batch.size(0))
        pre = x_batch[:cutoff].abs().mean(dim=(0, 1))
        std = x_batch.std(dim=(0, 1))
        score = pre / (std + 1e-6)
        topk = min(args.context_top_k, score.numel())
        context_indices = score.topk(topk).indices.tolist()
        print(f"[context] auto-selected indices: {context_indices}")

    row, model = run_prune_experiment(
        strategy="none",
        amount=0.0,
        no_prune=True,
        train_steps=args.train_steps,
        ft_steps=0,
        last_only=eval_last_only,
        eval_last_only=eval_last_only,
        seed=args.seed,
        device=args.device,
        movement_batches=20,
        model_type="ctrnn",
        hidden_size=args.hidden_size,
        task=args.task,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        mask_last_k=args.mask_last_k,
        mask_context_after_t=args.mask_context_after_t,
        context_indices=context_indices,
        save_model_path=str(checkpoint),
        run_id=args.run_id,
        return_model=True,
    )

    print("baseline_acc", row.get("post_acc"))
    if args.full_sequence:
        print("baseline_acc_sequence", row.get("post_acc_sequence"))

    with torch.no_grad():
        model.hidden_layer.weight.zero_()
        model.hidden_layer.bias.zero_()
        if args.strict_ablation:
            model.alpha = 1.0
            model.oneminusalpha = 0.0

    env_id = ensure_modcog_env_id(args.task)
    if env_id is None:
        raise ValueError(f"Unknown Mod_Cog task: {args.task}")

    criterion = torch.nn.CrossEntropyLoss()
    data = NeuroGymDatasetDM(
        env_id,
        T=args.ng_T,
        B=args.ng_B,
        device=args.device,
        last_only=eval_last_only,
        seed=args.seed,
        mask_last_k=args.mask_last_k,
        mask_context_after_t=args.mask_context_after_t,
        context_indices=context_indices,
    )

    metrics = evaluate(
        model,
        data,
        args.device,
        criterion,
        steps=args.eval_steps,
        dataset_last_only=eval_last_only,
        eval_last_only=eval_last_only,
    )

    print("ablated_acc", metrics["acc"])
    if args.full_sequence:
        print("ablated_acc_sequence", metrics["acc_sequence"])


if __name__ == "__main__":
    main()
