#!/usr/bin/env python3
"""Generate multiple CTRNN suite configs for stable Mod-Cog baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym

from pruning_benchmark.tasks.modcog import ensure_modcog_env_id, estimate_modcog_T, list_modcog_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5, help="Number of tasks to sample")
    parser.add_argument("--seed", type=int, default=7, help="Task sampling seed")
    parser.add_argument("--max_attempts", type=int, default=6, help="Max attempts to emit")
    parser.add_argument(
        "--output_dir",
        default="configs",
        help="Directory to write suite config JSON files",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints",
        help="Root directory for checkpoint folders",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _attempts() -> list[dict]:
    return [
        {"tag": "ctrnn_a1", "hidden": 256, "steps": 6000, "batch": 128, "lr": 1e-3, "clip": 1.0},
        {"tag": "ctrnn_a2", "hidden": 256, "steps": 6000, "batch": 256, "lr": 5e-4, "clip": 1.0},
        {"tag": "ctrnn_a3", "hidden": 384, "steps": 6000, "batch": 128, "lr": 1e-3, "clip": 0.5},
        {"tag": "ctrnn_a4", "hidden": 384, "steps": 8000, "batch": 128, "lr": 5e-4, "clip": 1.0},
        {"tag": "ctrnn_a5", "hidden": 256, "steps": 8000, "batch": 256, "lr": 7e-4, "clip": 1.0},
        {"tag": "ctrnn_a6", "hidden": 512, "steps": 6000, "batch": 128, "lr": 7e-4, "clip": 0.5},
    ]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    tasks = [t for t in list_modcog_tasks() if ("ctx" in t or "dlydm" in t)]
    if len(tasks) < args.count:
        raise SystemExit("Not enough ctx/dlydm tasks available.")
    selected = rng.sample(tasks, args.count)

    runs = []
    for name in selected:
        env_id = ensure_modcog_env_id(f"modcog:{name}")
        env = gym.make(env_id)
        T = estimate_modcog_T(env)
        runs.append(
            {
                "run_id": f"baseline_modcog_{name}_seed0",
                "strategy": "none",
                "amount": 0.0,
                "no_prune": True,
                "seed": 0,
                "task": f"modcog:{name}",
                "ng_T": int(T),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = _attempts()[: max(1, min(args.max_attempts, len(_attempts())))]

    for attempt in attempts:
        tag = attempt["tag"]
        suite_id = f"modcog_ctx_dlydm_random{args.count}_{tag}"
        ckpt_dir = Path(args.checkpoint_root) / suite_id
        output_csv = f"results/{suite_id}.csv"
        attempt_runs = []
        for run in runs:
            run = dict(run)
            task_name = run["task"].split("modcog:", 1)[-1]
            run["save_model_path"] = str(ckpt_dir / f"modcog_{task_name}_seed0.pt")
            attempt_runs.append(run)

        suite = {
            "run_id": suite_id,
            "output_csv": output_csv,
            "defaults": {
                "hidden_size": attempt["hidden"],
                "train_steps": attempt["steps"],
                "ft_steps": 0,
                "last_only": False,
                "device": args.device,
                "movement_batches": 20,
                "model_type": "ctrnn",
                "ng_T": 0,
                "ng_B": attempt["batch"],
                "eval_sample_batches": 32,
                "reset_results": False,
                "resume": True,
                "lr": attempt["lr"],
                "clip": attempt["clip"],
            },
            "runs": attempt_runs,
        }

        output_path = output_dir / f"{suite_id}.json"
        output_path.write_text(json.dumps(suite, indent=2) + "\n")
        print(f"Wrote {output_path}")

    print(f"Tasks: {', '.join(selected)}")


if __name__ == "__main__":
    main()
