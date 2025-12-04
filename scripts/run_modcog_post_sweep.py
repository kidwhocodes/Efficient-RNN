#!/usr/bin/env python3
"""Run post-training pruning methods on a subset of Mod_Cog tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pruning_benchmark.experiments import run_suite_from_config

DEFAULT_CONFIG_PATH = Path("configs/modcog_post_sweep.json")
DEFAULT_OUTPUT_CSV = "results/modcog_post_sweep.csv"
DEFAULT_TASKS = [
    "Go-v0",
    "Anti-v0",
    "RTGo-v0",
    "DelayGo-v0",
    "CtxDlyDM1-v0",
    "CtxDlyDM2-v0",
    "MultiDlyDM-v0",
    "DMS-v0",
    "DNMS-v0",
    "DMC-v0",
]
DEFAULT_STRATEGIES = ["random_unstructured", "l1_unstructured", "movement", "noise_prune"]
DEFAULT_AMOUNTS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_SEEDS = [0]


def _parse_list(src: str, cast=str):
    items = [item.strip() for item in src.split(",") if item.strip()]
    if not items:
        return []
    if cast is str:
        return items
    return [cast(item) for item in items]


def build_config(
    *,
    tasks: list[str],
    strategies: list[str],
    amounts: list[float],
    seeds: list[int],
    defaults: dict,
    run_id: str,
    output_csv: str,
) -> dict:
    runs: list[dict] = []
    for task in tasks:
        task_name = f"modcog:{task}"
        for seed in seeds:
            checkpoint = f"checkpoints/modcog_{task.replace('-', '_').lower()}_seed{seed}.pt"
            runs.append(
                {
                    "run_id": f"modcog_{task}_baseline_seed{seed}",
                    "strategy": "none",
                    "amount": 0.0,
                    "no_prune": True,
                    "seed": seed,
                    "task": task_name,
                    "train_steps": defaults["train_steps"],
                    "save_model_path": checkpoint,
                }
            )
            for strategy in strategies:
                for amount in amounts:
                    suffix = f"{task}_{strategy}_{int(amount * 100)}_seed{seed}"
                    runs.append(
                        {
                            "run_id": f"modcog_{suffix}",
                            "strategy": strategy,
                            "amount": amount,
                            "seed": seed,
                            "task": task_name,
                            "prune_phase": "post",
                            "skip_training": True,
                            "load_model_path": checkpoint,
                        }
                    )
    return {
        "run_id": run_id,
        "output_csv": output_csv,
        "defaults": defaults,
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run a Mod_Cog pruning suite.")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS), help="Comma-separated Mod_Cog task suffixes.")
    parser.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES), help="Comma-separated pruning strategies.")
    parser.add_argument("--amounts", type=str, default=",".join(str(a) for a in DEFAULT_AMOUNTS), help="Comma-separated pruning fractions.")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated integer seeds.")
    parser.add_argument("--run_id", type=str, default="modcog_post_sweep", help="Suite run identifier.")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to write the generated suite config.")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Where to collect run metrics.")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--ft_steps", type=int, default=10)
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--ng_T", type=int, default=600)
    parser.add_argument("--ng_B", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--last_only", dest="last_only", action="store_true", help="Train/eval using only final timestep (default).")
    parser.add_argument("--full_sequence", dest="last_only", action="store_false", help="Use full sequence loss.")
    parser.set_defaults(last_only=True)
    parser.add_argument("--reset_results", dest="reset_results", action="store_true", help="Clear previous run directories (default).")
    parser.add_argument("--keep_results", dest="reset_results", action="store_false", help="Append without resetting results.")
    parser.set_defaults(reset_results=True)
    parser.add_argument("--device_ft", type=str, default=None, help="Optional secondary device override.")
    return parser.parse_args()


def main() -> None:
    from scripts.register_modcog_envs import register_envs

    args = parse_args()
    tasks = _parse_list(args.tasks, str)
    strategies = _parse_list(args.strategies, str)
    amounts = _parse_list(args.amounts, float)
    seeds = _parse_list(args.seeds, int)
    defaults = {
        "hidden_size": args.hidden_size,
        "train_steps": args.train_steps,
        "ft_steps": args.ft_steps,
        "last_only": bool(args.last_only),
        "device": args.device,
        "movement_batches": args.movement_batches,
        "reset_results": bool(args.reset_results),
        "ng_T": args.ng_T,
        "ng_B": args.ng_B,
    }
    if args.device_ft:
        defaults["device_ft"] = args.device_ft

    cfg = build_config(
        tasks=tasks,
        strategies=strategies,
        amounts=amounts,
        seeds=seeds,
        defaults=defaults,
        run_id=args.run_id,
        output_csv=args.output_csv,
    )
    config_path = Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2))

    register_envs()
    csv_path = run_suite_from_config(str(config_path))
    print(f"[done] Suite results written to {csv_path}")


if __name__ == "__main__":
    main()
