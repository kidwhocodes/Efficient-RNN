#!/usr/bin/env python3
"""Generate a suite config that trains CTRNN baselines for all Mod-Cog tasks."""

import argparse
import json
from pathlib import Path

from pruning_benchmark.tasks.modcog import list_modcog_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="configs/modcog_all_baselines_1000.json",
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
        default="checkpoints/modcog_all_1000",
        help="Directory to save checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = list_modcog_tasks()
    if not tasks:
        raise SystemExit("No Mod-Cog tasks found. Is Mod_Cog installed?")

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
    for task_name in tasks:
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
        "run_id": "modcog_all_baselines_1000",
        "output_csv": "results/modcog_all_baselines_1000.csv",
        "defaults": defaults,
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"Wrote {output_path} with {len(runs)} runs")


if __name__ == "__main__":
    main()
