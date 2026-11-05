"""maxichem: VULCAN chemical networks, with photolysis, as a batched GPU 0-D solver for exoplanet GCMs (README.md).

The classes load on first use, so importing the package does not import torch:
    VulcanNetwork (kinvulcan), CudaSolver (kincuda), ADSolver (kincuda_ad).
"""

import importlib

_HOME = {"VulcanNetwork": "kinvulcan", "CudaSolver": "kincuda", "ADSolver": "kincuda_ad"}
__all__ = list(_HOME)


def __getattr__(name):
    if name in _HOME:
        return getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
