#!/usr/bin/env python3
"""Decision-level confusion analysis by task and trial condition for Mod-Cog pruning runs.

This script replays pre/post-pruning model behavior from a suite CSV and records
decision-level confusion counts. "Condition" is inferred from the underlying
Mod-Cog env trial metadata:

- if the trial dict exposes an explicit discrete field like `context`, `ctx`, or
  `rule`, that field is used;
- otherwise, a stable rounded signature of the `trial` dict is used.

Outputs:
1) raw confusion counts CSV
2) summary CSV with condition-level accuracy and top confusion
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments.runner import fresh_model, temporary_seed
from pruning_benchmark.analysis.replay import (
    resolve_modcog_B,
    resolve_modcog_T,
    resolve_noise_prune_kwargs,
    resolve_score_batch_setting,
)
from pruning_benchmark.pruning import PruneContext, get_pruner
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


def _to_int(v, default: int) -> int:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def _to_float(v, default: float) -> float:
    try:
        if v is None or v == "":
            return default
        out = float(v)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _infer_hidden_size(state: Dict[str, torch.Tensor]) -> int:
    if "hidden_layer.weight" in state:
        return int(state["hidden_layer.weight"].shape[0])
    for key in ("gru.weight_hh_l0", "lstm.weight_hh_l0", "rnn.weight_hh_l0"):
        if key in state:
            return int(state[key].shape[1])
    raise ValueError("Could not infer hidden size from checkpoint.")


def _extract_prune_kwargs(row: Dict[str, str], strategy: str, prune_seed: Optional[int]) -> Dict[str, object]:
    if strategy != "noise_prune":
        return {}
    return resolve_noise_prune_kwargs(row, cli_prune_seed=prune_seed)


def _sample_score_batches(
    data: ModCogTrialDM,
    num: int,
    device: str,
    max_resamples: int,
    min_valid: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(max(0, num)):
        chosen = None
        attempts = max(1, max_resamples)
        for _ in range(attempts):
            xb, yb = data.sample_batch()
            valid = int((yb >= 0).sum().item())
            if valid >= min_valid:
                chosen = (xb, yb)
                break
            if chosen is None:
                chosen = (xb, yb)
        xb, yb = chosen
        out.append((xb.to(device), yb.to(device)))
    return out


def _task_matches(task: str, include_substrs: Sequence[str], exclude_substrs: Sequence[str]) -> bool:
    low = task.lower()
    if include_substrs and not all(s in low for s in include_substrs):
        return False
    if exclude_substrs and any(s in low for s in exclude_substrs):
        return False
    return True


def _strategy_matches(strategy: str, keep: Optional[Sequence[str]]) -> bool:
    if not keep:
        return True
    return strategy in keep


def _condition_from_trial(trial: object) -> str:
    if not isinstance(trial, dict) or not trial:
        return "unknown"
    for key in ("context", "ctx", "rule", "modality", "choice", "cond"):
        if key in trial:
            value = trial[key]
            if isinstance(value, np.generic):
                value = value.item()
            return f"{key}={value}"
    parts = []
    for key in sorted(trial.keys()):
        value = trial[key]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 4)
        parts.append(f"{key}={value}")
    return "|".join(parts)


def _sample_eval_batches_with_conditions(
    data: ModCogTrialDM,
    eval_batches: int,
    device: str,
) -> List[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
    out: List[Tuple[torch.Tensor, torch.Tensor, List[str]]] = []
    T, B, I = data.T, data.B, data.input_dim
    for _ in range(max(1, eval_batches)):
        X = torch.zeros(T, B, I, device=device)
        Y = torch.full((T, B), -1, dtype=torch.long, device=device)
        conds: List[str] = []
        for b, env in enumerate(data.envs):
            ob, labels = data._sample_single(env)
            trial = getattr(getattr(env, "unwrapped", env), "trial", None)
            conds.append(_condition_from_trial(trial))
            t_len = min(T, ob.shape[0])
            if t_len > 0:
                X[:t_len, b] = torch.from_numpy(ob[:t_len]).float().to(device)
                Y[:t_len, b] = torch.from_numpy(labels[:t_len]).to(torch.long).to(device)
        out.append((X, Y, conds))
    return out


def _final_valid_label(column: torch.Tensor) -> Optional[int]:
    valid = torch.nonzero(column >= 0, as_tuple=False).view(-1)
    if valid.numel() == 0:
        return None
    return int(column[valid[-1]].item())


def _confusion_counts(
    model: torch.nn.Module,
    eval_batches: List[Tuple[torch.Tensor, torch.Tensor, List[str]]],
) -> Counter[Tuple[str, int, int]]:
    counts: Counter[Tuple[str, int, int]] = Counter()
    model.eval()
    with torch.no_grad():
        for xb, yb, conds in eval_batches:
            logits, _ = model(xb)
            pred = logits.argmax(dim=-1)
            for b, cond in enumerate(conds):
                true_label = _final_valid_label(yb[:, b])
                if true_label is None:
                    continue
                valid = torch.nonzero(yb[:, b] >= 0, as_tuple=False).view(-1)
                last_idx = int(valid[-1].item())
                pred_label = int(pred[last_idx, b].item())
                counts[(cond, true_label, pred_label)] += 1
    return counts


def _top_confusion(counts: Counter[Tuple[int, int]]) -> Tuple[str, int]:
    wrong = [(k, v) for k, v in counts.items() if k[0] != k[1]]
    if not wrong:
        return "none", 0
    (true_label, pred_label), count = max(wrong, key=lambda kv: kv[1])
    return f"{true_label}->{pred_label}", count


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision confusion by task/trial condition for pruning runs.")
    parser.add_argument("--suite_csv", required=True)
    parser.add_argument("--output_raw_csv", required=True)
    parser.add_argument("--output_summary_csv", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--score_batch_max_resamples", type=int, default=10)
    parser.add_argument("--score_batch_min_valid", type=int, default=1)
    parser.add_argument("--prune_rng_seed", type=int, default=None)
    parser.add_argument(
        "--task_include",
        default="",
        help="Comma-separated substrings all required in task.",
    )
    parser.add_argument(
        "--task_exclude",
        default="",
        help="Comma-separated substrings excluded from task.",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Optional comma-separated strategy whitelist.",
    )
    parser.add_argument(
        "--amounts",
        default="",
        help="Optional comma-separated amounts whitelist (e.g. 0.7,0.8,0.9).",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print progress every N processed rows (default: 10).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*env\\.gt.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*env\\.new_trial.*", category=UserWarning)

    include_substrs = tuple(s.strip().lower() for s in args.task_include.split(",") if s.strip())
    exclude_substrs = tuple(s.strip().lower() for s in args.task_exclude.split(",") if s.strip())
    keep_strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip()) or None
    keep_amounts = {
        round(float(s.strip()), 10)
        for s in args.amounts.split(",")
        if s.strip()
    } if args.amounts else None

    rows = list(csv.DictReader(Path(args.suite_csv).open()))
    selected_rows = []
    for row in rows:
        task = (row.get("task") or "").strip()
        strategy = (row.get("strategy") or "").strip()
        if not task.startswith("modcog:"):
            continue
        if not _task_matches(task, include_substrs, exclude_substrs):
            continue
        if not _strategy_matches(strategy, keep_strategies):
            continue
        amount = round(_to_float(row.get("amount"), 0.0), 10)
        if keep_amounts is not None and amount not in keep_amounts:
            continue
        ckpt = (row.get("load_model_path") or "").strip()
        if not ckpt or not Path(ckpt).exists():
            continue
        if ensure_modcog_env_id(task) is None:
            continue
        selected_rows.append(row)

    total_rows = len(selected_rows)
    if total_rows == 0:
        raise ValueError("No matching rows were found. Check filters and paths.")
    print(f"Processing {total_rows} rows with eval_batches={args.eval_batches}")

    raw_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(selected_rows, start=1):
        task = (row.get("task") or "").strip()
        strategy = (row.get("strategy") or "").strip()
        amount = _to_float(row.get("amount"), 0.0)
        seed = _to_int(row.get("seed"), 0)
        ckpt = (row.get("load_model_path") or "").strip()
        env_id = ensure_modcog_env_id(task)
        T = resolve_modcog_T(row, task)
        B = resolve_modcog_B(row)
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

        with temporary_seed(seed * 10 + 9):
            eval_batches = _sample_eval_batches_with_conditions(data, args.eval_batches, args.device)

        pre_counts = _confusion_counts(model, eval_batches)

        pruner = get_pruner(strategy)
        score_count = pruner.resolved_batch_count(_to_int(row.get("movement_batches"), 20))
        with temporary_seed(seed * 10 + 11):
            score_batches = _sample_score_batches(
                data,
                score_count,
                args.device,
                max_resamples=score_batch_max_resamples,
                min_valid=score_batch_min_valid,
            ) if score_count > 0 else None

        context = PruneContext(
            model=model,
            amount=amount,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=-1),
            last_only=False,
            device=args.device,
            batches=score_batches,
            metadata={"phase": "post", "run_id": row.get("run_id", "")},
        )
        prune_kwargs = _extract_prune_kwargs(row, strategy, args.prune_rng_seed)
        pruner.run(context, **prune_kwargs)
        post_counts = _confusion_counts(model, eval_batches)

        for phase, counts in (("pre", pre_counts), ("post", post_counts)):
            by_cond_pair: Dict[str, Counter[Tuple[int, int]]] = defaultdict(Counter)
            for (cond, true_label, pred_label), count in counts.items():
                by_cond_pair[cond][(true_label, pred_label)] += count
                raw_rows.append(
                    {
                        "task": task,
                        "strategy": strategy,
                        "amount": f"{amount:.1f}",
                        "phase": phase,
                        "condition_key": cond,
                        "true_label": str(true_label),
                        "pred_label": str(pred_label),
                        "count": str(count),
                    }
                )

            for cond, pair_counts in by_cond_pair.items():
                total = sum(pair_counts.values())
                correct = sum(v for (t, p), v in pair_counts.items() if t == p)
                top_err, top_err_count = _top_confusion(pair_counts)
                summary_rows.append(
                    {
                        "task": task,
                        "strategy": strategy,
                        "amount": f"{amount:.1f}",
                        "phase": phase,
                        "condition_key": cond,
                        "n_trials": str(total),
                        "acc": f"{correct / max(1, total):.8f}",
                        "top_confusion": top_err,
                        "top_confusion_count": str(top_err_count),
                    }
                )

        if not args.quiet:
            print(f"{task}\t{strategy}\t{amount:.1f}\tpre={len(pre_counts)} entries\tpost={len(post_counts)} entries")
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == total_rows):
            print(f"[progress] {idx}/{total_rows} ({100.0 * idx / total_rows:.1f}%) current={strategy}:{amount:.1f}:{task}")

    raw_path = Path(args.output_raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as fh:
        fields = ["task", "strategy", "amount", "phase", "condition_key", "true_label", "pred_label", "count"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Wrote {raw_path}")

    summary_path = Path(args.output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as fh:
        fields = ["task", "strategy", "amount", "phase", "condition_key", "n_trials", "acc", "top_confusion", "top_confusion_count"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
