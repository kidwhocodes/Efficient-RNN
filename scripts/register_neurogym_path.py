#!/usr/bin/env python3
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEURO = ROOT / "neurogym"
site = Path(sysconfig.get_paths()["purelib"])
site.mkdir(parents=True, exist_ok=True)
(site / "neurogym_local.pth").write_text(str(NEURO)+"\n")
print("[info] wrote", site / "neurogym_local.pth")
