#!/usr/bin/env python3
"""Print detailed Mod_Cog import diagnostics."""

import traceback

try:
    import Mod_Cog.mod_cog_tasks  # noqa: F401
except Exception:  # pragma: no cover
    print("[error] Mod_Cog import failed:")
    traceback.print_exc()
else:
    print("[info] Mod_Cog import succeeded.")
