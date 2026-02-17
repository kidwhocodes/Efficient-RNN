#!/usr/bin/env python3
import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_paths()["purelib"])
site_packages.mkdir(parents=True, exist_ok=True)
pth_path = site_packages / "neurogym_local.pth"
pth_path.write_text("/Users/sanjithsenthil/Efficient RNN Project/neurogym\n")
print(f"[info] wrote {pth_path}")
