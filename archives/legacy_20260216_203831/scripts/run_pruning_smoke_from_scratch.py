#!/usr/bin/env python3
"""Train a fresh model and run a small pruning sweep with ablation checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments.runner import append_results_csv, run_prune_experiment

PRETRAIN_STRATEGIES = {"snip", "synflow", "grasp"}


def _parse_list(src: str, cast):
    items = [item.strip() for item in src.split(",") if item.strip()]
    if cast is str:
        return items
    return [cast(item) for item in items]


def _task_slug(task: str) -> str:
    slug = task.lower().replace(":", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)


def ensure_modcog_registration(task: str) -> None:
    if task.startswith("modcog:"):
        from scripts.register_modcog_envs import register_envs

        register_envs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline model and run a small pruning smoke test.",
    )
    parser.add_argument("--task", default="synthetic")
    parser.add_argument(
        "--strategies",
        default="random_unstructured,l1_unstructured,noise_prune",
        help="Comma-separated pruning strategies to evaluate.",
    )
    parser.add_argument("--amounts", default="0.3,0.7", help="Comma-separated pruning fractions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=600)
    parser.add_argument("--ft_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_type", type=str, default="ctrnn")
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument(
        "--decision_delay",
        type=int,
        default=None,
        help="Override decision delay for synthetic tasks (higher = more memory).",
    )
    parser.add_argument(
        "--mask_last_k",
        type=int,
        default=None,
        help="Mask the last k timesteps in NeuroGym datasets to avoid trivial cues.",
    )
    parser.add_argument("--ng_T", type=int, default=None)
    parser.add_argument("--ng_B", type=int, default=None)
    parser.add_argument("--ng_kwargs", type=str, default=None)
    parser.add_argument("--ng_dataset_kwargs", type=str, default=None)
    parser.add_argument("--last_only", dest="last_only", action="store_true")
    parser.add_argument("--full_sequence", dest="last_only", action="store_false")
    parser.set_defaults(last_only=True)
    parser.add_argument("--run_id", type=str, default="scratch_prune_smoke")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_csv", type=str, default="results/scratch_prune_smoke.csv")
    parser.add_argument("--ablation_min_drop", type=float, default=0.3)
    parser.add_argument("--eval_steps_post", type=int, default=100)
    parser.add_argument("--eval_steps_pre", type=int, default=100)
    parser.add_argument("--eval_steps_post0", type=int, default=100)
    parser.add_argument("--eval_steps_pre0", type=int, default=50)
    return parser.parse_args()


def _build_defaults(args: argparse.Namespace) -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "hidden_size": args.hidden_size,
        "train_steps": args.train_steps,
        "ft_steps": args.ft_steps,
        "last_only": bool(args.last_only),
        "device": args.device,
        "model_type": args.model_type,
        "movement_batches": args.movement_batches,
        "eval_steps_pre0": args.eval_steps_pre0,
        "eval_steps_pre": args.eval_steps_pre,
        "eval_steps_post0": args.eval_steps_post0,
        "eval_steps_post": args.eval_steps_post,
    }
    if args.decision_delay is not None:
        defaults["decision_delay"] = args.decision_delay
    if args.mask_last_k is not None:
        defaults["mask_last_k"] = args.mask_last_k
    if args.ng_T is not None:
        defaults["ng_T"] = args.ng_T
    if args.ng_B is not None:
        defaults["ng_B"] = args.ng_B
    if args.ng_kwargs:
        defaults["ng_kwargs"] = args.ng_kwargs
    if args.ng_dataset_kwargs:
        defaults["ng_dataset_kwargs"] = args.ng_dataset_kwargs
    return defaults


def main() -> None:
    args = parse_args()
    ensure_modcog_registration(args.task)

    strategies = _parse_list(args.strategies, str)
    if not strategies:
        raise SystemExit("No strategies specified.")
    amounts = _parse_list(args.amounts, float)
    if not amounts:
        raise SystemExit("No pruning amounts specified.")

    slug = _task_slug(args.task)
    base_run_id = f"{args.run_id}_{slug}_seed{args.seed}"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{base_run_id}.pt"

    defaults = _build_defaults(args)
    rows: List[Dict[str, object]] = []

    baseline_row = run_prune_experiment(
        strategy="none",
        amount=0.0,
        no_prune=True,
        seed=args.seed,
        task=args.task,
        run_id=f"{base_run_id}_baseline",
        save_model_path=str(checkpoint_path),
        ablation_check=True,
        ablation_min_drop=args.ablation_min_drop,
        **defaults,
    )
    rows.append(baseline_row)

    for strategy in strategies:
        for amount in amounts:
            run_id = f"{base_run_id}_{strategy}_{int(amount * 100)}"
            kwargs = dict(defaults)
            kwargs.update(
                {
                    "strategy": strategy,
                    "amount": amount,
                    "seed": args.seed,
                    "task": args.task,
                    "run_id": run_id,
                }
            )
            if strategy in PRETRAIN_STRATEGIES:
                kwargs["prune_phase"] = "pre"
                kwargs["skip_training"] = False
            else:
                kwargs["prune_phase"] = "post"
                kwargs["skip_training"] = True
                kwargs["load_model_path"] = str(checkpoint_path)

            rows.append(run_prune_experiment(**kwargs))

    append_results_csv(rows, args.output_csv)
    print(f"[done] wrote {len(rows)} rows to {args.output_csv}")
    print(f"[checkpoint] {checkpoint_path}")
    print("[baseline]")
    print(json.dumps(
        {
            "post_acc": baseline_row.get("post_acc"),
            "ablation_post_acc": baseline_row.get("ablation_post_acc"),
            "ablation_drop": baseline_row.get("ablation_drop"),
            "ablation_pass": baseline_row.get("ablation_pass"),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
