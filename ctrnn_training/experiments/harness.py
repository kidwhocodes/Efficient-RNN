"""Configuration-driven evaluation harness for running pruning suites."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from ..utils import make_run_id
from .runner import append_results_csv, run_prune_experiment

SuiteConfig = Tuple[Dict, List[Dict], str, str]


def load_suite_config(path: str) -> SuiteConfig:
    """Load a JSON suite config returning (defaults, runs, output_csv, suite_id)."""
    with open(path, "r") as f:
        cfg = json.load(f)
    defaults = cfg.get("defaults", {})
    runs = cfg.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Suite config 'runs' must be a list of run specs.")
    output_csv = cfg.get("output_csv", "")
    suite_id = cfg.get("run_id", make_run_id("suite"))
    return defaults, runs, output_csv, suite_id


def run_suite_from_config(path: str) -> str:
    """
    Execute a suite described by a JSON config file.

    Returns the path to the CSV containing aggregated results (if any).
    """
    defaults, runs, output_csv, suite_id = load_suite_config(path)
    if len(runs) == 0:
        raise ValueError("Suite config contains no runs.")

    results_accum: List[Dict] = []
    csv_path = output_csv or os.path.join("results", f"{suite_id}.csv")
    Path(os.path.dirname(csv_path) or ".").mkdir(parents=True, exist_ok=True)

    reset_results = bool(defaults.pop("reset_results", False))
    if reset_results and os.path.exists(csv_path):
        os.remove(csv_path)

    base_models: Dict[Tuple, any] = {}
    quick_factor = defaults.pop("train_steps_factor", None)
    eval_steps_factor = defaults.pop("eval_steps_factor", None)

    for idx, spec in enumerate(runs, start=1):
        merged = {**defaults, **spec}
        run_id = merged.get("run_id") or f"{suite_id}_{idx}"
        merged["run_id"] = run_id
        if quick_factor is not None:
            if "train_steps" in merged:
                merged["train_steps"] = max(1, int(round(merged["train_steps"] * quick_factor)))
            if "ft_steps" in merged:
                merged["ft_steps"] = max(0, int(round(merged["ft_steps"] * quick_factor)))
        if eval_steps_factor is not None:
            for key_name, default_val in (
                ("eval_steps_pre0", 50),
                ("eval_steps_pre", 100),
                ("eval_steps_post0", 100),
                ("eval_steps_post", 100),
            ):
                base_val = merged.get(key_name, default_val)
                merged[key_name] = max(1, int(round(base_val * eval_steps_factor)))
        hidden_size = merged.get("hidden_size", defaults.get("hidden_size")) if isinstance(defaults, dict) else None
        key = (
            merged.get("task"),
            merged.get("seed"),
            hidden_size,
        )
        base_model = base_models.get(key)
        print(
            f"[suite:{suite_id}] ({idx}/{len(runs)}) running {merged['strategy']} "
            f"amount={merged.get('amount')} seed={merged.get('seed', 'NA')} run_id={run_id}"
        )
        row, model = run_prune_experiment(**merged, base_model=base_model, return_model=True)
        results_accum.append(row)
        append_results_csv([row], csv_path)
        if (merged.get("strategy") == "none" or merged.get("no_prune")) and model is not None:
            base_models[key] = model
    return csv_path


__all__ = ["load_suite_config", "run_suite_from_config"]
