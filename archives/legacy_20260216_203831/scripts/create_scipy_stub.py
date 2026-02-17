#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for module, body in {
    "optimize": "def curve_fit(*a, **k):\n    raise RuntimeError('scipy.optimize not available')\n",
    "special": "def erf(x):\n    return 0.0\n",
    "interpolate": "def interp1d(*a, **k):\n    raise RuntimeError('scipy.interpolate not available')\n",
}.items():
    pkg = ROOT / "scipy" / module
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(body)
print(f"[info] updated scipy stub under {ROOT/'scipy'}")
