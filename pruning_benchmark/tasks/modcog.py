"""Utilities for resolving Mod_Cog task builders."""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

ModCogBuilder = Callable[..., object]

_PREFIXES = (
    "mod_cog-",
    "modcog-",
    "mod_cog:",
    "modcog:",
    "modcog/",
)
_SUFFIXES = (
    "-v0",
    "-v1",
    "-v2",
    "_v0",
    "_v1",
    "_v2",
)


def _normalize_name(name: str) -> str:
    token = (name or "").strip().lower()
    if not token:
        return ""
    for prefix in _PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    for suffix in _SUFFIXES:
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    normalized = "".join(ch for ch in token if ch.isalnum())
    return normalized


@lru_cache(maxsize=1)
def _builder_registry() -> Dict[str, Tuple[str, ModCogBuilder]]:
    try:
        module = importlib.import_module("Mod_Cog.mod_cog_tasks")
    except ImportError as exc:  # pragma: no cover - handled by caller
        raise ImportError(
            "Mod_Cog tasks are unavailable. Install the Mod_Cog package or ensure it is on PYTHONPATH."
        ) from exc

    registry: Dict[str, Tuple[str, ModCogBuilder]] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        key = _normalize_name(name)
        if not key:
            continue
        registry[key] = (name, obj)
    return registry


def resolve_modcog_callable(name: str) -> Optional[Tuple[str, ModCogBuilder]]:
    """Return (canonical_name, builder) for a Mod_Cog task name, if available."""
    key = _normalize_name(name)
    if not key:
        return None
    registry = _builder_registry()
    return registry.get(key)


def list_modcog_tasks() -> Tuple[str, ...]:
    """List canonical Mod_Cog builder names."""
    registry = _builder_registry()
    names = sorted({canonical for canonical, _ in registry.values()})
    return tuple(names)


__all__ = ["resolve_modcog_callable", "list_modcog_tasks"]
