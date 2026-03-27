"""
Compat shim so `import neurogym` exposes the package bundled under `neurogym/neurogym`.

The upstream project places the actual package inside a nested `neurogym/`
directory and relies on namespace packages. Older code paths (including this
repo) import from `neurogym` directly, so we import the inner module and
re-export its namespace.
"""

from importlib import import_module as _import_module
import os as _os
import sys as _sys

_SB3_INSTALLED = False  # set early so nested imports can access it

_inner_root = _os.path.join(_os.path.dirname(__file__), "neurogym")
if _inner_root not in __path__:
    __path__.append(_inner_root)

_inner = _import_module(".neurogym", __name__)

# mirror the inner module namespace
globals().update({k: getattr(_inner, k) for k in dir(_inner) if not k.startswith("__") or k == "__all__"})
if hasattr(_inner, "__all__"):
    __all__ = list(_inner.__all__)

# expose common subpackages for absolute imports (e.g. neurogym.envs)
for _subpkg in ("envs", "utils", "wrappers", "config"):
    try:
        _module = _import_module(f".neurogym.{_subpkg}", __name__)
    except Exception:
        continue
    _sys.modules[f"{__name__}.{_subpkg}"] = _module
