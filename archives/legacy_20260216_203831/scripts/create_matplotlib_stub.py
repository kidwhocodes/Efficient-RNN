#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "matplotlib"

INIT = """rcParams = {}
from . import pyplot, animation, axes
"""

PYPLOT = """def subplots(*args, **kwargs):
    class DummyFig:
        def savefig(self, *a, **k): pass
        def tight_layout(self): pass
        def colorbar(self, *a, **k): pass
    class DummyAx:
        def plot(self, *a, **k): pass
        def set_title(self, *a, **k): pass
        def set_xlabel(self, *a, **k): pass
        def set_ylabel(self, *a, **k): pass
        def grid(self, *a, **k): pass
        def legend(self, *a, **k): pass
        def set_ylim(self, *a, **k): pass
    return DummyFig(), DummyAx()

def colorbar(*args, **kwargs): pass
def close(*args, **kwargs): pass
"""

ANIMATION = """class FuncAnimation:
    def __init__(self, *args, **kwargs): pass
    def save(self, *a, **k): pass
"""

AXES = """class Axes: pass
"""


def main() -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "__init__.py").write_text(INIT)
    (PKG / "pyplot.py").write_text(PYPLOT)
    (PKG / "animation.py").write_text(ANIMATION)
    (PKG / "axes.py").write_text(AXES)
    print(f"[info] wrote matplotlib stub to {PKG}")


if __name__ == "__main__":
    main()
