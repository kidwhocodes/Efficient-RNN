#!/usr/bin/env python3
"""Inspect raw Mod-Cog dataset targets for label validity and distribution."""

import argparse
import numpy as np

from pruning_benchmark.tasks.modcog import resolve_modcog_callable, estimate_modcog_T
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="modcog:ctxdlydm1")
    parser.add_argument("--T", type=int, default=0)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task
    if not task.startswith("modcog:"):
        raise SystemExit("Task must use modcog:<name> format")
    env_suffix = task.split("modcog:", 1)[1].strip()
    if not env_suffix:
        raise SystemExit("Missing Mod-Cog task suffix")

    builder_info = resolve_modcog_callable(env_suffix)
    if builder_info is None:
        raise SystemExit(f"Unknown Mod-Cog task: {task}")

    _, builder_fn = builder_info
    env = builder_fn()
    T = int(args.T) if args.T > 0 else estimate_modcog_T(env)
    B = int(args.B)

    dataset = ModCogTrialDM(
        env,
        T=T,
        B=B,
        device="cpu",
        last_only=False,
        seed=0,
        mask_fixation=True,
    )

    print(f"task={task} T={T} B={B}")
    for idx in range(args.samples):
        inputs, targets = dataset.sample_batch()
        y = np.asarray(targets)
        print(f"sample={idx} targets_shape={y.shape} dtype={y.dtype}")

        if y.ndim == 2:
            last = y[-1]
            print(
                "  last_step stats: min=%.3f max=%.3f nan=%.3f neg=%.3f unique=%s"
                % (
                    np.nanmin(last),
                    np.nanmax(last),
                    np.isnan(last).mean(),
                    (last < 0).mean(),
                    np.unique(last)[:10],
                )
            )
        elif y.ndim == 3:
            last = y[-1]
            sums = np.sum(last, axis=-1)
            print(
                "  last_step stats: min=%.3f max=%.3f nan=%.3f neg=%.3f"
                % (
                    np.nanmin(last),
                    np.nanmax(last),
                    np.isnan(last).mean(),
                    (last < 0).mean(),
                )
            )
            print(
                "  last_step sum stats: min=%.3f max=%.3f zero_frac=%.3f"
                % (np.min(sums), np.max(sums), (sums == 0).mean())
            )
        else:
            print("  unexpected target rank")

        trial_envs = getattr(dataset, "_trial_envs", None)
        if idx == 0 and trial_envs:
            env0 = trial_envs[0]
            start = getattr(env0, "start_ind", {}).get("decision")
            end = getattr(env0, "end_ind", {}).get("decision")
            gt = getattr(env0, "gt", None)
            print(f"  decision window: {start}..{end}")
            if gt is not None and start is not None and end is not None:
                dec = np.asarray(gt)[start:end]
                print(f"  env gt decision unique={np.unique(dec)[:10]}")


if __name__ == "__main__":
    main()
