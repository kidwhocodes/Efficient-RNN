#!/usr/bin/env python3
"""Generate a Mod-Cog ctx/dlydm CTRNN baseline suite with stronger defaults."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym

from pruning_benchmark.tasks.modcog import ensure_modcog_env_id, estimate_modcog_T, list_modcog_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="configs/modcog_ctx_dlydm_random5_ctrnn_suite.json",
        help="Path to write the suite config JSON",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/modcog_ctx_dlydm_random5_ctrnn",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--run_id", default="modcog_ctx_dlydm_random5_ctrnn_8000")
    parser.add_argument("--output_csv", default="results/modcog_ctx_dlydm_random5_ctrnn_8000.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    tasks = [t for t in list_modcog_tasks() if ("ctx" in t or "dlydm" in t)]
    if len(tasks) < args.count:
        raise SystemExit("Not enough ctx/dlydm tasks available.")
    selected = rng.sample(tasks, args.count)

    ckpt_dir = Path(args.checkpoint_dir)
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
                "save_model_path": str(ckpt_dir / f"modcog_{name}_seed0.pt"),
            }
        )

    suite = {
        "run_id": args.run_id,
        "output_csv": args.output_csv,
        "defaults": {
            "hidden_size": args.hidden_size,
            "train_steps": args.train_steps,
            "ft_steps": 0,
            "last_only": False,
            "device": args.device,
            "movement_batches": 20,
            "model_type": "ctrnn",
            "ng_T": 0,
            "ng_B": args.batch_size,
            "eval_sample_batches": 32,
            "reset_results": False,
            "resume": True,
            "lr": args.lr,
            "clip": args.clip,
        },
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"Wrote {output_path} with tasks: {', '.join(selected)}")


if __name__ == "__main__":
    main()
