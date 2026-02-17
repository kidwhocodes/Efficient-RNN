#!/usr/bin/env python3
"""Evaluate a checkpoint with the recurrent layer zeroed out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments import run_prune_experiment
from pruning_benchmark.tasks import (
    SynthCfg,
    SynthContextCfg,
    SynthHierContextCfg,
    SynthMultiRuleCfg,
    SynthNBackCfg,
    SyntheticContextDM,
    SyntheticDM,
    SyntheticHierContextDM,
    SyntheticMultiRuleDM,
    SyntheticNBackDM,
)
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import NeuroGymDatasetDM
from pruning_benchmark.training import evaluate


def _zero_hidden(model: torch.nn.Module) -> None:
    if not hasattr(model, "hidden_layer"):
        raise ValueError("Model has no hidden_layer to ablate.")
    with torch.no_grad():
        model.hidden_layer.weight.zero_()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate the recurrent layer and report accuracy.")
    parser.add_argument("--task", default="modcog:ctxdlydm1seql")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ng_T", type=int, default=400)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument(
        "--decision_delay",
        type=int,
        default=None,
        help="Override decision delay for synthetic tasks.",
    )
    parser.add_argument(
        "--full_sequence",
        action="store_true",
        help="Evaluate accuracy over the full sequence instead of last step only.",
    )
    args = parser.parse_args()

    run_id = f"hidden_ablation_check_{args.task.replace(':', '_').replace('/', '_')}"
    run_kwargs = {}
    if args.decision_delay is not None:
        run_kwargs["decision_delay"] = args.decision_delay

    row, model = run_prune_experiment(
        strategy="none",
        amount=0.0,
        train_steps=0,
        ft_steps=0,
        last_only=not args.full_sequence,
        eval_last_only=not args.full_sequence,
        seed=0,
        device=args.device,
        movement_batches=1,
        model_type=args.model_type,
        task=args.task,
        skip_training=True,
        load_model_path=args.checkpoint,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        hidden_size=args.hidden_size,
        return_model=True,
        run_id=run_id,
        **run_kwargs,
    )

    print("baseline_acc", row.get("post_acc"))
    if args.full_sequence:
        print("baseline_acc_sequence", row.get("post_acc_sequence"))

    _zero_hidden(model)
    dataset_task = args.task
    if dataset_task.startswith("modcog:"):
        env_id = ensure_modcog_env_id(dataset_task)
        if env_id is None:
            raise ValueError(f"Unknown Mod_Cog task: {dataset_task}")
        dataset_task = env_id

    if dataset_task == "synthetic":
        cfg = SynthCfg(decision_delay=args.decision_delay or SynthCfg.decision_delay)
        data = SyntheticDM(cfg)
    elif dataset_task == "synthetic_context":
        cfg = SynthContextCfg(decision_delay=args.decision_delay or SynthContextCfg.decision_delay)
        data = SyntheticContextDM(cfg)
    elif dataset_task == "synthetic_multirule":
        cfg = SynthMultiRuleCfg(decision_delay=args.decision_delay or SynthMultiRuleCfg.decision_delay)
        data = SyntheticMultiRuleDM(cfg)
    elif dataset_task == "synthetic_hiercontext":
        cfg = SynthHierContextCfg(decision_delay=args.decision_delay or SynthHierContextCfg.decision_delay)
        data = SyntheticHierContextDM(cfg)
    elif dataset_task == "synthetic_nback":
        cfg = SynthNBackCfg(decision_delay=args.decision_delay or SynthNBackCfg.decision_delay)
        data = SyntheticNBackDM(cfg)
    else:
        data = NeuroGymDatasetDM(
            dataset_task,
            T=args.ng_T,
            B=args.ng_B,
            device=args.device,
            last_only=not args.full_sequence,
            seed=0,
        )
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(
        model,
        data,
        args.device,
        criterion,
        steps=args.eval_steps,
        dataset_last_only=not args.full_sequence,
        eval_last_only=not args.full_sequence,
    )
    print("ablated_acc", metrics["acc"])
    if args.full_sequence:
        print("ablated_acc_sequence", metrics["acc_sequence"])


if __name__ == "__main__":
    main()
