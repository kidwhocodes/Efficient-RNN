"""Helpers for replaying pruning runs from suite CSV rows."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


MODCOG_T_BY_TASK = {
    "modcog:ctxdlydm1": 32,
    "modcog:ctxdlydm1intl": 32,
    "modcog:ctxdlydm1intr": 32,
    "modcog:ctxdlydm1intseq": 40,
    "modcog:ctxdlydm1seql": 40,
    "modcog:ctxdlydm1seqr": 40,
    "modcog:ctxdlydm2": 32,
    "modcog:ctxdlydm2intl": 32,
    "modcog:ctxdlydm2intr": 32,
    "modcog:ctxdlydm2intseq": 40,
    "modcog:ctxdlydm2seql": 40,
    "modcog:ctxdlydm2seqr": 40,
}


def to_int(value: object, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def to_float(value: object, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def to_bool(value: object, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def resolve_modcog_T(row: Dict[str, str], task: str, default: int = 40) -> int:
    logged = to_int(row.get("ng_T"), 0)
    if logged > 0:
        return logged
    if task in MODCOG_T_BY_TASK:
        return MODCOG_T_BY_TASK[task]
    if task.startswith("modcog:"):
        return 40 if "seq" in task.lower() else 32
    return default


def resolve_modcog_B(row: Dict[str, str], default: int = 256) -> int:
    logged = to_int(row.get("ng_B"), 0)
    return logged if logged > 0 else default


def resolve_row_seed(row: Dict[str, str], override_seed: Optional[int] = None) -> int:
    if override_seed is not None:
        return int(override_seed)
    return to_int(row.get("seed"), 0)


def resolve_score_batch_setting(
    row: Dict[str, str],
    key: str,
    default: int,
) -> int:
    logged = to_int(row.get(key), 0)
    return logged if logged > 0 else int(default)


def resolve_noise_prune_kwargs(
    row: Dict[str, str],
    *,
    cli_prune_seed: Optional[int] = None,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "sigma": to_float(row.get("noise_sigma", row.get("prune_sigma")), 1.0),
        "eps": to_float(row.get("noise_eps", row.get("prune_eps")), 0.3),
        "leak_shift": to_float(row.get("noise_leak_shift", row.get("prune_leak_shift")), 0.0),
        "matched_diagonal": to_bool(
            row.get("noise_matched_diagonal", row.get("prune_matched_diagonal")),
            True,
        ),
    }
    rescale = row.get("prune_rescale_weights")
    if rescale not in {None, ""}:
        kwargs["rescale_weights"] = to_bool(rescale, True)

    seed = cli_prune_seed
    if seed is None:
        for key in ("noise_rng_seed", "prune_rng_seed", "rng_seed"):
            value = row.get(key)
            if value not in {None, ""}:
                seed = to_int(value, 0)
                break
    if seed is not None:
        kwargs["rng"] = np.random.default_rng(int(seed))
    return kwargs
