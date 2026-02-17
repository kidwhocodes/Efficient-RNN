#!/usr/bin/env python3
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "Mod_Cog"
site = Path(sysconfig.get_paths()["purelib"])
site.mkdir(parents=True, exist_ok=True)
(site / "mod_cog.pth").write_text(str(MOD)+"\n")
print("[info] wrote", site / "mod_cog.pth")
