#!/usr/bin/env python3
"""Verify LSTM pruning by comparing reported vs actual recurrent sparsity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.analysis.metrics import recurrent_sparsity
from pruning_benchmark.experiments import run_prune_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LSTM recurrent pruning sparsity.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="modcog:ctxdlydm1seql")
    parser.add_argument("--amount", type=float, default=0.7)
    parser.add_argument("--ng_T", type=int, default=400)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden_size", type=int, default=None)
    args = parser.parse_args()

    row, model = run_prune_experiment(
        strategy="l1_unstructured",
        amount=args.amount,
        train_steps=0,
        ft_steps=0,
        skip_training=True,
        load_model_path=args.checkpoint,
        task=args.task,
        model_type="lstm",
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        device=args.device,
        hidden_size=args.hidden_size,
        return_model=True,
    )

    print("reported_post_sparsity_recurrent", row.get("post_sparsity_recurrent"))
    print("actual_recurrent_sparsity", recurrent_sparsity(model))


if __name__ == "__main__":
    main()
