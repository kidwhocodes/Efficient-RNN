#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = ROOT / "pydantic"
pkg.mkdir(parents=True, exist_ok=True)
(pkg / "__init__.py").write_text("""class BaseModel: pass\nclass PositiveInt(int): pass\nfield_validator = lambda *a, **k: (lambda f: f)\nField = lambda *a, **k: None\n""")
print(f"[info] wrote pydantic stub to {pkg}")
