#!/usr/bin/env python3
"""Helper script to run a Mod_Cog NeuroGym pruning experiment."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pruning on a Mod_Cog NeuroGym task.")
    parser.add_argument(
        "--env",
        default="FlexibleWorkingMemory-v0",
        help="Mod_Cog environment suffix (without the Mod_Cog- prefix).",
    )
    parser.add_argument("--strategy", default="l1_unstructured", help="Pruning strategy name.")
    parser.add_argument("--amount", type=float, default=0.2, help="Pruning fraction (0-1).")
    parser.add_argument("--train_steps", type=int, default=200, help="Baseline training steps.")
    parser.add_argument("--ft_steps", type=int, default=0, help="Fine-tuning steps after pruning.")
    parser.add_argument("--ng_T", type=int, default=600, help="Sequence length sampled from Mod_Cog Dataset.")
    parser.add_argument("--ng_B", type=int, default=32, help="Batch size sampled from Mod_Cog Dataset.")
    parser.add_argument(
        "--ng_kwargs",
        type=str,
        default=None,
        help="JSON string forwarded to the Mod_Cog environment constructor.",
    )
    parser.add_argument(
        "--ng_dataset_kwargs",
        type=str,
        default=None,
        help="JSON string forwarded to neurogym.Dataset (e.g., {\"max_batch\": 500}).",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device.")
    args = parser.parse_args()

    try:
        import Mod_Cog.mod_cog_tasks  # noqa: F401
    except ImportError as exc:  # pragma: no cover - user environment
        raise SystemExit(
            "Mod_Cog is not importable. Install it via `pip install -e /path/to/Mod_Cog` first."
        ) from exc

    from pruning_benchmark.experiments import run_prune_experiment

    def _parse_json(blob: str | None) -> dict | None:
        if not blob:
            return None
        return json.loads(blob)

    task = f"modcog:{args.env}"
    row = run_prune_experiment(
        strategy=args.strategy,
        amount=args.amount,
        train_steps=args.train_steps,
        ft_steps=args.ft_steps,
        device=args.device,
        task=task,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        ng_kwargs=_parse_json(args.ng_kwargs),
        ng_dataset_kwargs=_parse_json(args.ng_dataset_kwargs),
        skip_training=False,
    )
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
