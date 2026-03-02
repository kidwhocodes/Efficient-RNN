#!/usr/bin/env python3
"""Generate Mod-Cog family baseline suites for multiple model types."""

import argparse
import json
import re
from pathlib import Path

from pruning_benchmark.tasks.modcog import list_modcog_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="configs",
        help="Directory to write suite config JSONs",
    )
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--ng_T", type=int, default=0)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--eval_sample_batches", type=int, default=32)
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/modcog_family_1000",
        help="Root directory to save checkpoints per model type",
    )
    parser.add_argument(
        "--max_families",
        type=int,
        default=0,
        help="Optional cap on number of families (0 = no cap).",
    )
    parser.add_argument(
        "--models",
        default="ctrnn,rnn",
        help="Comma-separated model types to generate (ctrnn,rnn,gru,lstm)",
    )
    return parser.parse_args()


def family_key(name: str) -> str:
    token = name.strip().lower()
    base = token.split("_", 1)[0]
    base = re.sub(r"\d+$", "", base)
    return base or token


def select_tasks(max_families: int) -> list[str]:
    tasks = list_modcog_tasks()
    if not tasks:
        raise SystemExit("No Mod-Cog tasks found. Is Mod_Cog installed?")
    families = {}
    for task in tasks:
        key = family_key(task)
        families.setdefault(key, []).append(task)
    selected = []
    for key in sorted(families.keys()):
        selected.append(sorted(families[key])[0])
        if max_families and len(selected) >= max_families:
            break
    return selected


def write_suite(output_path: Path, model_type: str, tasks: list[str], args: argparse.Namespace) -> None:
    defaults = {
        "hidden_size": args.hidden_size,
        "train_steps": int(args.train_steps),
        "ft_steps": 0,
        "last_only": True,
        "device": args.device,
        "movement_batches": 20,
        "ng_T": args.ng_T,
        "ng_B": args.ng_B,
        "eval_sample_batches": args.eval_sample_batches,
        "ablation_check": True,
        "ablation_min_drop": 0.0,
        "reset_results": False,
        "resume": True,
        "model_type": model_type,
    }

    runs = []
    ckpt_dir = Path(args.checkpoint_root) / model_type
    for task_name in tasks:
        run_id = f"baseline_modcog_{task_name}_{model_type}_seed{args.seed}"
        checkpoint_path = ckpt_dir / f"modcog_{task_name}_{model_type}_seed{args.seed}.pt"
        runs.append(
            {
                "run_id": run_id,
                "strategy": "none",
                "amount": 0.0,
                "no_prune": True,
                "seed": int(args.seed),
                "task": f"modcog:{task_name}",
                "save_model_path": str(checkpoint_path),
            }
        )

    suite = {
        "run_id": f"modcog_family_baselines_1000_{model_type}",
        "output_csv": f"results/modcog_family_baselines_1000_{model_type}.csv",
        "defaults": defaults,
        "runs": runs,
    }

    output_path.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"Wrote {output_path} with {len(runs)} runs")


def main() -> None:
    args = parse_args()
    tasks = select_tasks(args.max_families)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        output_path = output_dir / f"modcog_family_baselines_1000_{model}.json"
        write_suite(output_path, model, tasks, args)


if __name__ == "__main__":
    main()
