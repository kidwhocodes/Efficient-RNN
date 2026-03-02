#!/usr/bin/env python3
"""Generate a smaller Mod-Cog baseline suite (one task per family)."""

import argparse
import json
import re
from pathlib import Path

from pruning_benchmark.tasks.modcog import list_modcog_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="configs/modcog_family_baselines_1000.json",
        help="Path to write the suite config JSON",
    )
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--ng_T", type=int, default=0)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--eval_sample_batches", type=int, default=32)
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/modcog_family_1000",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--max_families",
        type=int,
        default=0,
        help="Optional cap on number of families (0 = no cap).",
    )
    return parser.parse_args()


def family_key(name: str) -> str:
    """Best-effort family grouping by stripping trailing digits/suffixes."""
    token = name.strip().lower()
    # Split on underscore first if present.
    base = token.split("_", 1)[0]
    # Strip trailing digits.
    base = re.sub(r"\d+$", "", base)
    # If we stripped everything, fall back to token.
    return base or token


def main() -> None:
    args = parse_args()
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
        if args.max_families and len(selected) >= args.max_families:
            break

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
    }

    runs = []
    ckpt_dir = Path(args.checkpoint_dir)
    for task_name in selected:
        run_id = f"baseline_modcog_{task_name}_seed{args.seed}"
        checkpoint_path = ckpt_dir / f"modcog_{task_name}_seed{args.seed}.pt"
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
        "run_id": "modcog_family_baselines_1000",
        "output_csv": "results/modcog_family_baselines_1000.csv",
        "defaults": defaults,
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"Wrote {output_path} with {len(runs)} runs")


if __name__ == "__main__":
    main()
