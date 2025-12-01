#!/usr/bin/env python3
"""List registered Mod_Cog gym environments."""

import gym

try:
    import Mod_Cog.mod_cog_tasks  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Mod_Cog is not importable. Install it via `pip install -e /path/to/Mod_Cog`."
    ) from exc

def main() -> None:
    env_registry = getattr(gym.envs, "registry", None)
    if env_registry is None:
        iterable = getattr(gym.envs, "values", lambda: [])()
    else:
        iterable = env_registry.values()
    ids = [spec.id for spec in iterable if spec.id.startswith("Mod_Cog")]
    print(f"Found {len(ids)} Mod_Cog environments.")
    for name in sorted(ids):
        print(name)


if __name__ == "__main__":
    main()
