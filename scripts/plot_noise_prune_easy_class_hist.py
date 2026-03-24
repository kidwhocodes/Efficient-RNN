#!/usr/bin/env python3
"""Plot average pre/post class-prediction histogram for noise-prune on easy Mod-Cog tasks."""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.analysis.replay import (
    resolve_modcog_B,
    resolve_modcog_T,
    resolve_noise_prune_kwargs,
    resolve_score_batch_setting,
)
from pruning_benchmark.experiments.runner import fresh_model, temporary_seed
from pruning_benchmark.pruning import PruneContext, get_pruner
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


def _to_int(value, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _to_float(value, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _infer_hidden_size(state: Dict[str, torch.Tensor]) -> int:
    if "hidden_layer.weight" in state:
        return int(state["hidden_layer.weight"].shape[0])
    for key in ("gru.weight_hh_l0", "lstm.weight_hh_l0", "rnn.weight_hh_l0"):
        if key in state:
            return int(state[key].shape[1])
    raise ValueError("Could not infer hidden size from checkpoint.")


def _is_easy_task(task: str) -> bool:
    low = task.lower()
    return ("ctx" in low and "dly" in low and "dm" in low and "seq" not in low)


def _extract_noise_kwargs(row: Dict[str, str]) -> Dict[str, object]:
    return resolve_noise_prune_kwargs(row)


def _sample_score_batches(
    data: ModCogTrialDM,
    num: int,
    device: str,
    max_resamples: int,
    min_valid: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(max(0, num)):
        chosen_x = None
        chosen_y = None
        for _ in range(max(1, max_resamples)):
            xb, yb = data.sample_batch()
            valid = int((yb >= 0).sum().item())
            if valid >= min_valid:
                chosen_x, chosen_y = xb, yb
                break
            if chosen_x is None:
                chosen_x, chosen_y = xb, yb
        batches.append((chosen_x.to(device), chosen_y.to(device)))
    return batches


def _prediction_histogram(
    model: torch.nn.Module,
    eval_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    n_classes: int,
) -> torch.Tensor:
    counts = torch.zeros(n_classes, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for xb, yb in eval_batches:
            logits, _ = model(xb)
            pred = logits.argmax(dim=-1)
            valid = yb >= 0
            if valid.any():
                binc = torch.bincount(pred[valid].reshape(-1), minlength=n_classes).to(torch.float64)
                counts += binc
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise-prune easy-task pre/post class histogram.")
    parser.add_argument("--suite_csv", required=True)
    parser.add_argument("--output_plot", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--score_batch_max_resamples", type=int, default=10)
    parser.add_argument("--score_batch_min_valid", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--amount",
        type=float,
        default=0.9,
        help="Noise-prune amount to compare against pre-prune baseline (default: 0.9).",
    )
    parser.add_argument(
        "--prune_rng_seed",
        type=int,
        default=None,
        help="Optional fixed RNG seed for deterministic noise-prune masks.",
    )
    parser.add_argument(
        "--num_prune_seeds",
        type=int,
        default=1,
        help="Number of prune RNG seeds to average over (default: 1).",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*env\\.gt.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*env\\.new_trial.*", category=UserWarning)

    rows = list(csv.DictReader(Path(args.suite_csv).open()))
    selected = []
    for r in rows:
        if (r.get("strategy") or "").strip() != "noise_prune":
            continue
        if not _is_easy_task((r.get("task") or "").strip()):
            continue
        amount = _to_float(r.get("amount"), -1.0)
        if abs(amount - float(args.amount)) > 1e-9:
            continue
        selected.append(r)
    if not selected:
        raise ValueError(f"No noise_prune rows for easy tasks found at amount={args.amount}.")

    pre_total = None
    post_totals: List[torch.Tensor] = []
    task_counts = defaultdict(int)

    for row in selected:
        task = (row.get("task") or "").strip()
        ckpt = (row.get("load_model_path") or "").strip()
        if not task or not ckpt or not Path(ckpt).exists():
            continue
        env_id = ensure_modcog_env_id(task)
        if env_id is None:
            continue
        T = resolve_modcog_T(row, task)
        B = resolve_modcog_B(row)
        seed = _to_int(row.get("seed"), 0)
        score_batch_max_resamples = resolve_score_batch_setting(
            row, "score_batch_max_resamples", args.score_batch_max_resamples
        )
        score_batch_min_valid = resolve_score_batch_setting(
            row, "score_batch_min_valid", args.score_batch_min_valid
        )
        data = ModCogTrialDM(
            env_id,
            T=T,
            B=B,
            device=args.device,
            last_only=False,
            seed=seed,
            mask_fixation=True,
        )

        state = torch.load(ckpt, map_location=args.device)
        hidden = _infer_hidden_size(state)
        model_type = (row.get("model_type") or "ctrnn").strip()
        model = fresh_model(
            input_dim=data.input_dim,
            hidden_size=hidden,
            output_dim=data.n_classes,
            device=args.device,
            model_type=model_type,
        )
        model.load_state_dict(state)

        # Reuse fixed eval batches for fair pre/post comparison.
        with temporary_seed(seed * 10 + 9):
            eval_batches = [(xb.to(args.device), yb.to(args.device)) for xb, yb in (data.sample_batch() for _ in range(max(1, args.eval_batches)))]

        pre_hist = _prediction_histogram(model, eval_batches, data.n_classes)

        pre_total = pre_hist if pre_total is None else (pre_total + pre_hist)

        # Evaluate post-prune histograms across multiple prune RNG seeds.
        seed_hists: List[torch.Tensor] = []
        for seed_idx in range(max(1, int(args.num_prune_seeds))):
            # Rebuild model from checkpoint so each seed starts from same baseline.
            model_seed = fresh_model(
                input_dim=data.input_dim,
                hidden_size=hidden,
                output_dim=data.n_classes,
                device=args.device,
                model_type=model_type,
            )
            model_seed.load_state_dict(state)

            pruner = get_pruner("noise_prune")
            score_count = pruner.resolved_batch_count(_to_int(row.get("movement_batches"), 20))
            with temporary_seed(seed * 10 + 11 + seed_idx):
                score_batches = _sample_score_batches(
                    data,
                    score_count,
                    args.device,
                    max_resamples=score_batch_max_resamples,
                    min_valid=score_batch_min_valid,
                ) if score_count > 0 else None

            context = PruneContext(
                model=model_seed,
                amount=_to_float(row.get("amount"), 0.0),
                criterion=torch.nn.CrossEntropyLoss(ignore_index=-1),
                last_only=False,
                device=args.device,
                batches=score_batches,
                metadata={"phase": "post", "run_id": row.get("run_id", "")},
            )
            prune_kwargs = _extract_noise_kwargs(row)
            if args.prune_rng_seed is not None:
                base_seed = int(args.prune_rng_seed)
            else:
                base_seed = _to_int(row.get("noise_rng_seed"), 0)
            if base_seed > 0 or args.prune_rng_seed is not None:
                prune_kwargs["rng"] = np.random.default_rng(base_seed + seed_idx)
            pruner.run(context, **prune_kwargs)
            seed_hists.append(_prediction_histogram(model_seed, eval_batches, data.n_classes))

        post_totals.extend(seed_hists)
        task_counts[task] += 1
        if not args.quiet:
            print(f"[ok] {task} amount={row.get('amount')} checkpoint={ckpt}")

    if pre_total is None or not post_totals:
        raise ValueError("No valid rows could be evaluated.")

    pre_freq = pre_total / max(float(pre_total.sum().item()), 1.0)
    post_freqs = []
    for post_total in post_totals:
        post_freqs.append(post_total / max(float(post_total.sum().item()), 1.0))
    post_stack = torch.stack(post_freqs, dim=0)
    post_freq_mean = post_stack.mean(dim=0)
    post_freq_std = post_stack.std(dim=0, unbiased=False)
    classes = list(range(len(pre_freq)))

    out = Path(args.output_plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    width = 0.42
    xs = [c for c in classes]
    plt.bar([x - width / 2 for x in xs], pre_freq.tolist(), width=width, label="pre-prune", alpha=0.85)
    plt.bar(
        [x + width / 2 for x in xs],
        post_freq_mean.tolist(),
        width=width,
        yerr=post_freq_std.tolist(),
        capsize=3,
        label=f"post-noise-prune (mean±std, n={len(post_totals)})",
        alpha=0.85,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Frequency")
    plt.title("Noise-prune (easy tasks): average predicted-class histogram (pre vs post)")
    plt.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

    print(f"Wrote {out}")
    print(f"amount={args.amount}")
    print(f"num_prune_seeds={args.num_prune_seeds}")
    print(f"post_hist_samples={len(post_totals)}")
    print("easy tasks evaluated:", dict(task_counts))


if __name__ == "__main__":
    main()
