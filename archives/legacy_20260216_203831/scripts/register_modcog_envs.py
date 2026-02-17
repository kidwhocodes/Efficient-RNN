#!/usr/bin/env python3
"""Register a curated subset of Mod_Cog tasks with Gym/NeuroGym."""

from __future__ import annotations

try:
    import gym
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Gym is required to register Mod_Cog environments.") from exc


def _env_registered(env_id: str) -> bool:
    registry = getattr(gym.envs, "registry", None)
    if registry is None:
        return False
    try:
        if hasattr(registry, "__contains__"):
            return env_id in registry  # Gym >=0.26 maps __contains__ to EnvRegistry
    except TypeError:
        pass
    all_envs = getattr(registry, "keys", None)
    if callable(all_envs):
        try:
            return env_id in all_envs()
        except TypeError:
            return env_id in list(all_envs())
    return False

try:
    import Mod_Cog.mod_cog_tasks as mod_cog
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Mod_Cog is not importable. Ensure the local repo is registered via scripts/register_modcog_path.py"
    ) from exc

ENV_FUNCTIONS = {
    "Go-v0": "go",
    "Anti-v0": "anti",
    "RTGo-v0": "rtgo",
    "DelayGo-v0": "dlygo",
    "CtxDlyDM1-v0": "ctxdlydm1",
    "CtxDlyDM2-v0": "ctxdlydm2",
    "MultiDlyDM-v0": "multidlydm",
    "DMS-v0": "dms",
    "DNMS-v0": "dnms",
    "DMC-v0": "dmc",
}


def register_envs() -> None:
    count = 0
    for suffix, func_name in ENV_FUNCTIONS.items():
        env_id = f"Mod_Cog-{suffix}"
        if _env_registered(env_id):
            continue
        entry_point = f"Mod_Cog.mod_cog_tasks:{func_name}"
        gym.register(id=env_id, entry_point=entry_point)
        count += 1
    print(f"[info] Registered {count} Mod_Cog environments")


def main() -> None:
    register_envs()


if __name__ == "__main__":
    main()
