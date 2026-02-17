#!/usr/bin/env python3
"""Compare task/alpha variants to isolate which change drives easy performance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments import run_prune_experiment


@dataclass(frozen=True)
class Variant:
    task: str
    dt: float
    tau: float


VARIANTS = [
    Variant("modcog:ctxdm1", 20.0, 100.0),
    Variant("modcog:ctxdm1", 100.0, 100.0),
    Variant("modcog:ctxdlydm1seql", 20.0, 100.0),
    Variant("modcog:ctxdlydm1seql", 100.0, 100.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation matrix for task/alpha effects.")
    parser.add_argument("--checkpoint_ctxdm1", required=True)
    parser.add_argument("--checkpoint_ctxdlydm1seql", required=True)
    parser.add_argument("--ng_T", type=int, default=400)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for variant in VARIANTS:
        ckpt = (
            args.checkpoint_ctxdlydm1seql
            if "ctxdlydm1seql" in variant.task
            else args.checkpoint_ctxdm1
        )
        row = run_prune_experiment(
            strategy="none",
            amount=0.0,
            train_steps=0,
            ft_steps=0,
            last_only=True,
            seed=0,
            device=args.device,
            movement_batches=1,
            task=variant.task,
            skip_training=True,
            load_model_path=ckpt,
            ng_T=args.ng_T,
            ng_B=args.ng_B,
            dt=variant.dt,
            tau=variant.tau,
            run_id=f"ablate_{variant.task}_{variant.dt}_{variant.tau}",
        )
        print(
            f"{variant.task} dt={variant.dt} tau={variant.tau} "
            f"pre_acc={row.get('pre_acc')} post0_acc={row.get('post0_acc')}"
        )


if __name__ == "__main__":
    main()
