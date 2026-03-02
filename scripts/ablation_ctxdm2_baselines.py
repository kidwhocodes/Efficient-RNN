#!/usr/bin/env python3
"""Report ablated (zero recurrent) accuracy for ctxdm2 baseline checkpoints."""

import argparse
from pathlib import Path

from pruning_benchmark.experiments.runner import run_prune_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints_ctxdm2_fresh",
        help="Directory containing modcog_ctxdm2_seed*.pt",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated baseline seeds to evaluate",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_steps_post", type=int, default=100)
    parser.add_argument("--eval_steps_pre", type=int, default=100)
    parser.add_argument("--eval_steps_pre0", type=int, default=50)
    parser.add_argument("--eval_steps_post0", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ckpt_dir = Path(args.checkpoint_dir)

    print("seed\tcheckpoint\tablation_post_acc\tpost_acc")
    for seed in seeds:
        ckpt = ckpt_dir / f"modcog_ctxdm2_seed{seed}.pt"
        if not ckpt.exists():
            print(f"{seed}\t{ckpt}\tMISSING\tMISSING")
            continue
        row = run_prune_experiment(
            strategy="none",
            amount=0.0,
            train_steps=0,
            ft_steps=0,
            last_only=True,
            seed=seed,
            device=args.device,
            task="modcog:ctxdm2",
            no_prune=True,
            skip_training=True,
            load_model_path=str(ckpt),
            ablation_check=True,
            eval_steps_pre0=args.eval_steps_pre0,
            eval_steps_pre=args.eval_steps_pre,
            eval_steps_post0=args.eval_steps_post0,
            eval_steps_post=args.eval_steps_post,
        )
        ablated = row.get("ablation_post_acc")
        post_acc = row.get("post_acc")
        ablated_str = f"{ablated:.6f}" if isinstance(ablated, float) else str(ablated)
        post_str = f"{post_acc:.6f}" if isinstance(post_acc, float) else str(post_acc)
        print(f"{seed}\t{ckpt}\t{ablated_str}\t{post_str}")


if __name__ == "__main__":
    main()
