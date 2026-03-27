#!/usr/bin/env python3
"""Compare one-shot vs iterative prune+fine-tune on two Mod-Cog tasks for several methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments.runner import append_results_csv, run_prune_experiment


DEFAULT_TASKS: Dict[str, Tuple[int, str]] = {
    "modcog:ctxdlydm2intl": (
        32,
        "checkpoints/modcog_ctxdlydm12_ctrnn_a5/modcog_ctxdlydm2intl_seed0.pt",
    ),
    "modcog:ctxdlydm2intseq": (
        40,
        "checkpoints/modcog_ctxdlydm12_ctrnn_a5/modcog_ctxdlydm2intseq_seed0.pt",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--easy-task", default="modcog:ctxdlydm2intl")
    p.add_argument("--hard-task", default="modcog:ctxdlydm2intseq")
    p.add_argument(
        "--strategies",
        default="noise_prune,l1_unstructured,obd,random_unstructured",
    )
    p.add_argument("--ft-steps", type=int, default=100)
    p.add_argument("--eval-sample-batches", type=int, default=32)
    p.add_argument("--movement-batches", type=int, default=20)
    p.add_argument("--ng-b", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", default="results/modcog_iterative_prune_compare.csv")
    p.add_argument(
        "--output-plot",
        default="plots/modcog_iterative_prune_compare/post_acc_sequence_compare.png",
    )
    p.add_argument("--reset", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _task_info(task: str) -> Tuple[int, str]:
    if task not in DEFAULT_TASKS:
        raise ValueError(f"Unsupported task {task!r}; add it to DEFAULT_TASKS first.")
    return DEFAULT_TASKS[task]


def _extra_prune_kwargs(strategy: str, stage_idx: int) -> dict:
    if strategy == "noise_prune":
        return {"noise_rng_seed": stage_idx}
    return {}


def _append_row(rows: List[dict], regime: str, stage_idx: int, amount: float, row: dict) -> None:
    record = dict(row)
    record["pilot_regime"] = regime
    record["pilot_stage"] = stage_idx
    record["pilot_cumulative_amount"] = amount
    rows.append(record)


def _base_kwargs(task: str, ng_T: int, args: argparse.Namespace, strategy: str, stage_idx: int) -> dict:
    return dict(
        strategy=strategy,
        train_steps=0,
        ft_steps=args.ft_steps,
        last_only=False,
        model_type="ctrnn",
        eval_last_only=False,
        seed=args.seed,
        device=args.device,
        movement_batches=args.movement_batches,
        task=task,
        skip_training=True,
        ng_T=ng_T,
        ng_B=args.ng_b,
        eval_sample_batches=args.eval_sample_batches,
        lr=args.lr,
        clip=args.clip,
        hidden_size=256,
        return_model=True,
        **_extra_prune_kwargs(strategy, stage_idx),
    )


def run_one_shot(task: str, ng_T: int, ckpt: str, strategy: str, args: argparse.Namespace) -> List[dict]:
    rows: List[dict] = []
    for stage_idx in range(1, 10):
        amount = stage_idx / 10.0
        kwargs = _base_kwargs(task, ng_T, args, strategy, stage_idx)
        kwargs.update(
            amount=amount,
            run_id=f"pilot_{strategy}_oneshot_{task.replace('modcog:', '')}_p{stage_idx * 10}",
            load_model_path=ckpt,
        )
        row, _ = run_prune_experiment(**kwargs)
        _append_row(rows, "one_shot", stage_idx, amount, row)
    return rows


def run_iterative(task: str, ng_T: int, ckpt: str, strategy: str, args: argparse.Namespace) -> List[dict]:
    rows: List[dict] = []
    model = None
    for stage_idx in range(1, 10):
        amount = stage_idx / 10.0
        kwargs = _base_kwargs(task, ng_T, args, strategy, stage_idx)
        kwargs.update(
            amount=amount,
            run_id=f"pilot_{strategy}_iterative_{task.replace('modcog:', '')}_p{stage_idx * 10}",
        )
        if model is None:
            kwargs["load_model_path"] = ckpt
        else:
            kwargs["base_model"] = model
        row, model = run_prune_experiment(**kwargs)
        _append_row(rows, "iterative", stage_idx, amount, row)
    return rows


def make_plot(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r["task"] for r in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        subset = [r for r in rows if r["task"] == task]
        keys = sorted({(r["strategy"], r["pilot_regime"]) for r in subset})
        for strategy, regime in keys:
            pts = sorted(
                (
                    float(r["pilot_cumulative_amount"]),
                    float(r["post_acc_sequence"]),
                )
                for r in subset
                if r["strategy"] == strategy and r["pilot_regime"] == regime
            )
            label = f"{strategy}:{regime}"
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=label)
        ax.set_title(task)
        ax.set_xlabel("cumulative prune amount")
        ax.set_ylabel("post_acc_sequence")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    tasks = [args.easy_task, args.hard_task]
    rows: List[dict] = []
    for task in tasks:
        ng_T, ckpt = _task_info(task)
        for strategy in strategies:
            if not args.quiet:
                print(f"[pilot] task={task} strategy={strategy} regime=one_shot")
            rows.extend(run_one_shot(task, ng_T, ckpt, strategy, args))
            if not args.quiet:
                print(f"[pilot] task={task} strategy={strategy} regime=iterative")
            rows.extend(run_iterative(task, ng_T, ckpt, strategy, args))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and out_csv.exists():
        out_csv.unlink()
    append_results_csv(rows, str(out_csv))
    make_plot(rows, Path(args.output_plot))
    print(f"wrote {out_csv}")
    print(f"wrote {args.output_plot}")


if __name__ == "__main__":
    main()
