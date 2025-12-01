#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = ROOT / "pydantic_settings"
pkg.mkdir(parents=True, exist_ok=True)
(pkg / "__init__.py").write_text("""class BaseSettings: pass\ndef SettingsConfigDict(*args, **kwargs): return {}
""")
print(f"[info] wrote pydantic_settings stub to {pkg}")
