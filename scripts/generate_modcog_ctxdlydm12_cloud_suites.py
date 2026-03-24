#!/usr/bin/env python3
"""Generate deterministic multi-seed cloud suites for the CTRNN Mod-Cog benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TASKS: Tuple[Dict[str, object], ...] = (
    {"task": "modcog:ctxdlydm1", "ng_T": 32, "split": "easy_nonseq"},
    {"task": "modcog:ctxdlydm1intl", "ng_T": 32, "split": "easy_nonseq"},
    {"task": "modcog:ctxdlydm1intr", "ng_T": 32, "split": "easy_nonseq"},
    {"task": "modcog:ctxdlydm1intseq", "ng_T": 40, "split": "hard_seq"},
    {"task": "modcog:ctxdlydm1seql", "ng_T": 40, "split": "hard_seq"},
    {"task": "modcog:ctxdlydm1seqr", "ng_T": 40, "split": "hard_seq"},
    {"task": "modcog:ctxdlydm2", "ng_T": 32, "split": "easy_nonseq"},
    {"task": "modcog:ctxdlydm2intl", "ng_T": 32, "split": "easy_nonseq"},
    {"task": "modcog:ctxdlydm2intr", "ng_T": 32, "split": "easy_nonseq"},
    {"task": "modcog:ctxdlydm2intseq", "ng_T": 40, "split": "hard_seq"},
    {"task": "modcog:ctxdlydm2seql", "ng_T": 40, "split": "hard_seq"},
    {"task": "modcog:ctxdlydm2seqr", "ng_T": 40, "split": "hard_seq"},
)
TASK_INDEX_BY_TASK = {str(task_info["task"]): index for index, task_info in enumerate(TASKS)}

RAW_METHODS: Tuple[str, ...] = (
    "noise_prune",
    "l1_unstructured",
    "obd",
    "random_unstructured",
)

RAW_AMOUNTS: Tuple[float, ...] = tuple(round(step / 10.0, 1) for step in range(1, 10))
FT_AMOUNTS: Tuple[float, ...] = (0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output_dir", default="configs/cloud")
    parser.add_argument("--results_dir", default="results/cloud")
    parser.add_argument("--checkpoint_dir", default="checkpoints/cloud")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=8000)
    parser.add_argument("--ft_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--movement_batches", type=int, default=20)
    parser.add_argument("--eval_sample_batches", type=int, default=32)
    parser.add_argument("--noise_sigma", type=float, default=1.0)
    parser.add_argument("--noise_eps", type=float, default=0.3)
    parser.add_argument("--noise_leak_shift", type=float, default=0.0)
    parser.add_argument("--score_batch_max_resamples", type=int, default=10)
    parser.add_argument("--score_batch_min_valid", type=int, default=1)
    return parser.parse_args()


def _parse_ints(src: str) -> Tuple[int, ...]:
    items = tuple(int(part.strip()) for part in src.split(",") if part.strip())
    if not items:
        raise ValueError("At least one seed is required.")
    return items


def _slug(task: str) -> str:
    return task.replace("modcog:", "")


def _amount_tag(amount: float) -> str:
    return f"p{int(round(amount * 100.0))}"


def _seed_tag(seeds: Sequence[int]) -> str:
    return f"{len(seeds)}seed"


def _noise_rng_seed(seed: int, task_index: int, amount: float) -> int:
    amount_bucket = int(round(amount * 10.0))
    return int(seed) * 100000 + int(task_index) * 100 + amount_bucket


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _base_defaults(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "hidden_size": int(args.hidden_size),
        "train_steps": int(args.train_steps),
        "ft_steps": 0,
        "last_only": False,
        "device": str(args.device),
        "movement_batches": int(args.movement_batches),
        "model_type": "ctrnn",
        "ng_T": 0,
        "ng_B": int(args.batch_size),
        "eval_sample_batches": int(args.eval_sample_batches),
        "reset_results": False,
        "resume": True,
        "lr": float(args.lr),
        "clip": float(args.clip),
    }


def _prune_defaults(args: argparse.Namespace, *, ft_steps: int) -> Dict[str, object]:
    defaults = _base_defaults(args)
    defaults.update(
        {
            "train_steps": 0,
            "ft_steps": int(ft_steps),
            "skip_training": True,
            "prune_phase": "post",
            "noise_sigma": float(args.noise_sigma),
            "noise_eps": float(args.noise_eps),
            "noise_leak_shift": float(args.noise_leak_shift),
            "noise_matched_diagonal": 1,
            "score_batch_max_resamples": int(args.score_batch_max_resamples),
            "score_batch_min_valid": int(args.score_batch_min_valid),
        }
    )
    return defaults


def _baseline_runs(
    tasks: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    checkpoint_root: Path,
) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for task_info in tasks:
        task = str(task_info["task"])
        ng_t = int(task_info["ng_T"])
        slug = _slug(task)
        for seed in seeds:
            runs.append(
                {
                    "run_id": f"baseline_{slug}_seed{seed}",
                    "strategy": "none",
                    "amount": 0.0,
                    "no_prune": True,
                    "seed": int(seed),
                    "task": task,
                    "ng_T": ng_t,
                    "save_model_path": str(checkpoint_root / f"{slug}_seed{seed}.pt"),
                }
            )
    return runs


def _prune_runs(
    tasks: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    methods: Sequence[str],
    amounts: Sequence[float],
    baseline_root: Path,
    *,
    save_root: Path | None = None,
) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for task_info in tasks:
        task = str(task_info["task"])
        task_index = TASK_INDEX_BY_TASK[task]
        ng_t = int(task_info["ng_T"])
        slug = _slug(task)
        for seed in seeds:
            load_path = baseline_root / f"{slug}_seed{seed}.pt"
            for method in methods:
                for amount in amounts:
                    run: Dict[str, object] = {
                        "run_id": f"{method}_{slug}_seed{seed}_{_amount_tag(amount)}",
                        "strategy": method,
                        "amount": float(amount),
                        "seed": int(seed),
                        "task": task,
                        "ng_T": ng_t,
                        "load_model_path": str(load_path),
                    }
                    if method == "noise_prune":
                        run["noise_rng_seed"] = _noise_rng_seed(seed, task_index, amount)
                    if save_root is not None:
                        run["save_model_path"] = str(
                            save_root / f"{method}_{slug}_seed{seed}_{_amount_tag(amount)}.pt"
                        )
                    runs.append(run)
    return runs


def _subset(tasks: Sequence[Dict[str, object]], split: str) -> List[Dict[str, object]]:
    return [dict(task_info) for task_info in tasks if task_info["split"] == split]


def main() -> None:
    args = parse_args()
    seeds = _parse_ints(args.seeds)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    seed_tag = _seed_tag(seeds)
    ft_tag = f"ft{int(args.ft_steps)}"

    baseline_checkpoint_root = checkpoint_dir / f"modcog_ctxdlydm12_ctrnn_baselines_{seed_tag}"

    baseline_config_path = output_dir / f"modcog_ctxdlydm12_ctrnn_cloud_baselines_{seed_tag}.json"
    baseline_suite = {
        "run_id": f"modcog_ctxdlydm12_ctrnn_cloud_baselines_{seed_tag}",
        "output_csv": str(results_dir / f"modcog_ctxdlydm12_ctrnn_cloud_baselines_{seed_tag}.csv"),
        "defaults": _base_defaults(args),
        "runs": _baseline_runs(TASKS, seeds, baseline_checkpoint_root),
    }
    _write_json(baseline_config_path, baseline_suite)

    manifest: Dict[str, object] = {
        "seeds": list(seeds),
        "methods": list(RAW_METHODS),
        "raw_amounts": list(RAW_AMOUNTS),
        "ft_amounts": list(FT_AMOUNTS),
        "primary_metric": "post_acc_sequence",
        "stratification": "task_family",
        "noise_rng_seed_scheme": "seed*100000 + global_task_index*100 + amount_bucket",
        "notes": [
            "CTRNN-only benchmark.",
            "Mod-Cog runs use last_only=false and explicit ng_T per task.",
            "Raw post-prune suites omit unnecessary checkpoint saving.",
            "Fine-tune suites save final post-prune/post-FT checkpoints.",
            "Only top-4 comparison methods are included in the primary cloud benchmark.",
        ],
        "tasks": list(TASKS),
        "task_splits": {
            "easy_nonseq": [task_info["task"] for task_info in TASKS if task_info["split"] == "easy_nonseq"],
            "hard_seq": [task_info["task"] for task_info in TASKS if task_info["split"] == "hard_seq"],
        },
        "run_order": [str(baseline_config_path)],
        "suites": {},
    }
    manifest["suites"]["baseline"] = {
        "config_path": str(baseline_config_path),
        "output_csv": baseline_suite["output_csv"],
        "run_count": len(baseline_suite["runs"]),
    }

    raw_paths: List[str] = []
    ft_paths: List[str] = []
    total_runs = len(baseline_suite["runs"])

    for split in ("easy_nonseq", "hard_seq"):
        tasks = _subset(TASKS, split)

        raw_config_path = output_dir / f"modcog_ctxdlydm12_ctrnn_cloud_raw_top4_{split}_{seed_tag}.json"
        raw_suite = {
            "run_id": f"modcog_ctxdlydm12_ctrnn_cloud_raw_top4_{split}_{seed_tag}",
            "output_csv": str(results_dir / f"modcog_ctxdlydm12_ctrnn_cloud_raw_top4_{split}_{seed_tag}.csv"),
            "defaults": _prune_defaults(args, ft_steps=0),
            "runs": _prune_runs(tasks, seeds, RAW_METHODS, RAW_AMOUNTS, baseline_checkpoint_root),
        }
        _write_json(raw_config_path, raw_suite)
        raw_paths.append(str(raw_config_path))
        manifest["suites"][f"raw_{split}"] = {
            "config_path": str(raw_config_path),
            "output_csv": raw_suite["output_csv"],
            "run_count": len(raw_suite["runs"]),
        }
        total_runs += len(raw_suite["runs"])

        ft_checkpoint_root = checkpoint_dir / f"modcog_ctxdlydm12_ctrnn_{ft_tag}_top4_{split}_{seed_tag}"
        ft_config_path = output_dir / f"modcog_ctxdlydm12_ctrnn_cloud_{ft_tag}_top4_{split}_{seed_tag}.json"
        ft_suite = {
            "run_id": f"modcog_ctxdlydm12_ctrnn_cloud_{ft_tag}_top4_{split}_{seed_tag}",
            "output_csv": str(results_dir / f"modcog_ctxdlydm12_ctrnn_cloud_{ft_tag}_top4_{split}_{seed_tag}.csv"),
            "defaults": _prune_defaults(args, ft_steps=int(args.ft_steps)),
            "runs": _prune_runs(
                tasks,
                seeds,
                RAW_METHODS,
                FT_AMOUNTS,
                baseline_checkpoint_root,
                save_root=ft_checkpoint_root,
            ),
        }
        _write_json(ft_config_path, ft_suite)
        ft_paths.append(str(ft_config_path))
        manifest["suites"][f"{ft_tag}_{split}"] = {
            "config_path": str(ft_config_path),
            "output_csv": ft_suite["output_csv"],
            "run_count": len(ft_suite["runs"]),
        }
        total_runs += len(ft_suite["runs"])

    manifest["run_order"] = [str(baseline_config_path), *raw_paths, *ft_paths]
    manifest["total_run_count"] = total_runs

    manifest_path = output_dir / f"modcog_ctxdlydm12_ctrnn_cloud_manifest_{seed_tag}.json"
    _write_json(manifest_path, manifest)

    print(f"Wrote {baseline_config_path}")
    for key in ("raw_easy_nonseq", "raw_hard_seq", f"{ft_tag}_easy_nonseq", f"{ft_tag}_hard_seq"):
        print(f"Wrote {manifest['suites'][key]['config_path']}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
