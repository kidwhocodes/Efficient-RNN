#!/usr/bin/env python3
"""Print baseline and recurrent-ablated accuracy for Mod-Cog checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments import run_prune_experiment
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM
from pruning_benchmark.training import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report baseline vs. recurrent-ablated accuracy for Mod-Cog tasks."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/modcog_ctxdm1_ctxdm2",
        help="Directory containing modcog_<task>_seed<seed>.pt checkpoints.",
    )
    parser.add_argument(
        "--tasks",
        default="ctxdm1,ctxdm2",
        help="Comma-separated Mod-Cog task names (without modcog: prefix).",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds to evaluate.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ng_T", type=int, default=400)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument(
        "--full_sequence",
        action="store_true",
        help="Evaluate accuracy over the full sequence instead of last step only.",
    )
    return parser.parse_args()


def _zero_hidden(model: torch.nn.Module) -> None:
    if not hasattr(model, "hidden_layer"):
        raise ValueError("Model has no hidden_layer to ablate.")
    with torch.no_grad():
        model.hidden_layer.weight.zero_()


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ckpt_dir = Path(args.checkpoint_dir)
    last_only = not args.full_sequence

    header = ["task", "seed", "checkpoint", "post_acc", "ablated_acc"]
    if args.full_sequence:
        header.extend(["post_acc_sequence", "ablated_acc_sequence"])
    print("\t".join(header))

    for task in tasks:
        task_name = f"modcog:{task}"
        for seed in seeds:
            ckpt = ckpt_dir / f"modcog_{task}_seed{seed}.pt"
            if not ckpt.exists():
                print(f"{task}\t{seed}\t{ckpt}\tMISSING\tMISSING")
                continue

            eval_last_only = last_only
            dataset_last_only = last_only
            if task_name.startswith("modcog:"):
                last_only = False
                eval_last_only = False
                dataset_last_only = False
            row, model = run_prune_experiment(
                strategy="none",
                amount=0.0,
                train_steps=0,
                ft_steps=0,
                last_only=last_only,
                eval_last_only=eval_last_only,
                seed=seed,
                device=args.device,
                movement_batches=1,
                model_type=args.model_type,
                task=task_name,
                skip_training=True,
                load_model_path=str(ckpt),
                ng_T=args.ng_T,
                ng_B=args.ng_B,
                hidden_size=args.hidden_size,
                return_model=True,
                run_id=f"ablation_check_{task}_{seed}",
            )
            post_acc = row.get("post_acc")
            post_seq_acc = row.get("post_acc_sequence")

            _zero_hidden(model)
            env_id = ensure_modcog_env_id(task_name)
            if env_id is None:
                raise ValueError(f"Unknown Mod_Cog task: {task_name}")
            data = ModCogTrialDM(
                env_id,
                T=args.ng_T,
                B=args.ng_B,
                device=args.device,
                last_only=dataset_last_only,
                seed=seed,
                mask_fixation=True,
            )
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
            metrics = evaluate(
                model,
                data,
                args.device,
                criterion,
                steps=args.eval_steps,
                dataset_last_only=dataset_last_only,
                eval_last_only=eval_last_only,
            )
            ablated_acc = metrics["acc"]
            ablated_seq_acc = metrics.get("acc_sequence")

            fields = [
                task,
                str(seed),
                str(ckpt),
                f"{post_acc:.6f}" if isinstance(post_acc, float) else str(post_acc),
                f"{ablated_acc:.6f}" if isinstance(ablated_acc, float) else str(ablated_acc),
            ]
            if args.full_sequence:
                fields.extend(
                    [
                        f"{post_seq_acc:.6f}"
                        if isinstance(post_seq_acc, float)
                        else str(post_seq_acc),
                        f"{ablated_seq_acc:.6f}"
                        if isinstance(ablated_seq_acc, float)
                        else str(ablated_seq_acc),
                    ]
                )
            print("\t".join(fields))


if __name__ == "__main__":
    main()
