#!/usr/bin/env python3
"""Correlate analysis metrics with pruning performance from a suite CSV.

Supports metric CSVs produced by the analysis scripts (grouped or per-task).
By default, performance is mean post_acc_sequence grouped by strategy/amount.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _f(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _group_mean(
    rows: Iterable[Dict[str, str]],
    key_fields: Sequence[str],
    value_field: str,
) -> Dict[Tuple[str, ...], float]:
    grouped: Dict[Tuple[str, ...], List[float]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        v = _f(row.get(value_field))
        if v is None:
            continue
        grouped.setdefault(key, []).append(v)
    return {k: float(np.mean(vs)) for k, vs in grouped.items() if vs}


def _infer_metric_column(header: Sequence[str]) -> Optional[str]:
    preferred_suffix = (
        "_mean",
        "post_over_pre",
        "delta_mle",
        "post_mle",
        "acc_drop",
        "acc_ratio",
        "rho_pres_mean",
        "effective_connectivity",
        "edge_density",
        "mean_abs_weight_nz",
        "giant_component_frac",
    )
    for col in header:
        low = col.lower()
        if low in {"amount", "n", "strategy", "task"}:
            continue
        if any(low.endswith(s) or low == s for s in preferred_suffix):
            return col
    # fallback: first non-key numeric-looking column name
    for col in header:
        if col.lower() not in {"amount", "n", "strategy", "task"}:
            return col
    return None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _build_performance_map(rows: List[Dict[str, str]], perf_metric: str) -> Dict[Tuple[str, str], float]:
    if perf_metric == "delta_post_acc_sequence":
        # Derived because legacy rows may not contain this explicitly.
        values: Dict[Tuple[str, str], List[float]] = {}
        for row in rows:
            pre = _f(row.get("pre_acc_sequence"))
            post = _f(row.get("post_acc_sequence"))
            if pre is None or post is None:
                continue
            key = (str(row.get("strategy", "")), str(row.get("amount", "")))
            values.setdefault(key, []).append(pre - post)
        return {k: float(np.mean(v)) for k, v in values.items() if v}
    return _group_mean(rows, ("strategy", "amount"), perf_metric)


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate analysis metrics with pruning performance.")
    parser.add_argument("--suite_csv", required=True, help="Pruning suite CSV (e.g. results/..._prune.csv)")
    parser.add_argument(
        "--metric_csvs",
        nargs="+",
        required=True,
        help="One or more analysis CSVs to correlate against performance.",
    )
    parser.add_argument(
        "--perf_metric",
        default="post_acc_sequence",
        choices=("post_acc_sequence", "post_loss", "delta_post_acc_sequence"),
        help="Performance field from suite CSV (or derived sequence drop).",
    )
    parser.add_argument(
        "--metric_column",
        default=None,
        help="Optional fixed metric column name for all metric CSVs (auto-infer by default).",
    )
    parser.add_argument(
        "--invert_perf",
        action="store_true",
        help="Invert performance sign before correlation (useful for post_loss).",
    )
    args = parser.parse_args()

    suite_rows = _read_csv(Path(args.suite_csv))
    perf_map = _build_performance_map(suite_rows, args.perf_metric)
    if not perf_map:
        raise ValueError(f"No usable performance rows found for metric '{args.perf_metric}'.")

    print("metric_source\tmetric_column\tn_pairs\tpearson\tspearman")
    for metric_path in args.metric_csvs:
        path = Path(metric_path)
        if not path.exists():
            print(f"[skip] missing metric csv: {path}")
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        header = list(rows[0].keys())
        metric_col = args.metric_column or _infer_metric_column(header)
        if metric_col is None:
            print(f"{path}\t<none>\t0\tnan\tnan")
            continue

        metric_map = _group_mean(rows, ("strategy", "amount"), metric_col)
        xs: List[float] = []
        ys: List[float] = []
        for key, metric_val in metric_map.items():
            perf_val = perf_map.get(key)
            if perf_val is None:
                continue
            xs.append(metric_val)
            ys.append(-perf_val if args.invert_perf else perf_val)

        if len(xs) < 2:
            print(f"{path}\t{metric_col}\t{len(xs)}\tnan\tnan")
            continue

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        pear = _pearson(x, y)
        spear = _spearman(x, y)
        print(f"{path}\t{metric_col}\t{len(xs)}\t{pear:.6f}\t{spear:.6f}")


if __name__ == "__main__":
    main()
