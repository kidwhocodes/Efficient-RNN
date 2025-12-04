#!/usr/bin/env python3
"""Run every pruning method on a single task (e.g., one Mod_Cog env)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from pruning_benchmark.experiments import run_suite_from_config

DEFAULT_STRATEGIES = [
    "random_unstructured",
    "l1_unstructured",
    "movement",
    "snip",
    "synflow",
    "fisher",
    "grasp",
    "obd",
    "set",
    "obs",
    "woodfisher",
    "causal",
    "noise_prune",
]

PRETRAIN_STRATEGIES = {"snip", "synflow", "grasp"}


def _parse_list(src: str, cast):
    items = [item.strip() for item in src.split(",") if item.strip()]
    if cast is str:
        return items
    return [cast(item) for item in items]


def _task_slug(task: str) -> str:
    slug = task.lower().replace(":", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)


def build_runs(
    task: str,
    checkpoint: str,
    strategies: List[str],
    amounts: List[float],
    seed: int,
    train_steps: int,
) -> List[Dict]:
    runs: List[Dict] = []
    runs.append(
        {
            "run_id": f"{_task_slug(task)}_baseline_seed{seed}",
            "strategy": "none",
            "amount": 0.0,
            "no_prune": True,
            "seed": seed,
            "task": task,
            "save_model_path": checkpoint,
            "train_steps": train_steps,
        }
    )
    for strat in strategies:
        for amount in amounts:
            suffix = f"{_task_slug(task)}_{strat}_{int(amount * 100)}_seed{seed}"
            entry: Dict[str, object] = {
                "run_id": suffix,
                "strategy": strat,
                "amount": amount,
                "seed": seed,
                "task": task,
            }
            if strat in PRETRAIN_STRATEGIES:
                entry["prune_phase"] = "pre"
                entry["skip_training"] = False
                entry["train_steps"] = train_steps
            else:
                entry["prune_phase"] = "post"
                entry["skip_training"] = True
                entry["load_model_path"] = checkpoint
            runs.append(entry)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every pruning method on a single task.")
    parser.add_argument("--task", default="modcog:Go-v0", help="Task identifier (e.g., modcog:Go-v0).")
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated pruning strategies (defaults to all built-ins).",
    )
    parser.add_argument(
        "--amounts",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated pruning fractions to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=600)
    parser.add_argument("--ft_steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--ng_T", type=int, default=None)
    parser.add_argument("--ng_B", type=int, default=None)
    parser.add_argument("--ng_kwargs", type=str, default=None)
    parser.add_argument("--ng_dataset_kwargs", type=str, default=None)
    parser.add_argument("--last_only", dest="last_only", action="store_true")
    parser.add_argument("--full_sequence", dest="last_only", action="store_false")
    parser.set_defaults(last_only=True)
    parser.add_argument("--config_path", type=str, default="configs/single_task_all_pruners.json")
    parser.add_argument("--output_csv", type=str, default="results/single_task_all_pruners.csv")
    parser.add_argument("--run_id", type=str, default="single_task_all_pruners")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--reset_results", dest="reset_results", action="store_true")
    parser.add_argument("--keep_results", dest="reset_results", action="store_false")
    parser.set_defaults(reset_results=True)
    return parser.parse_args()


def ensure_modcog_registration(task: str) -> None:
    if task.startswith("modcog:"):
        from scripts.register_modcog_envs import register_envs

        register_envs()


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
    checkpoint = f"{args.checkpoint_dir}/{slug}_seed{args.seed}.pt"
    runs = build_runs(args.task, checkpoint, strategies, amounts, args.seed, args.train_steps)
    defaults: Dict[str, object] = {
        "hidden_size": args.hidden_size,
        "train_steps": args.train_steps,
        "ft_steps": args.ft_steps,
        "last_only": bool(args.last_only),
        "device": args.device,
        "movement_batches": args.movement_batches,
        "reset_results": bool(args.reset_results),
    }
    if args.ng_T is not None:
        defaults["ng_T"] = args.ng_T
    if args.ng_B is not None:
        defaults["ng_B"] = args.ng_B
    if args.ng_kwargs:
        defaults["ng_kwargs"] = args.ng_kwargs
    if args.ng_dataset_kwargs:
        defaults["ng_dataset_kwargs"] = args.ng_dataset_kwargs

    cfg = {
        "run_id": args.run_id,
        "output_csv": args.output_csv,
        "defaults": defaults,
        "runs": runs,
    }
    config_path = Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2))

    csv_path = run_suite_from_config(str(config_path))
    print(f"[done] All results stored in {csv_path}")


if __name__ == "__main__":
    main()
