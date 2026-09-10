"""Optional ML tools, imported lazily so analytic workflows need no PyTorch."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DeepEnsemble",
    "SplitConformal",
    "SplitConformalRegressor",
    "TransportBundle",
    "TransportMLP",
]


def __getattr__(name):
    modules = {
        "DeepEnsemble": "ensemble",
        "SplitConformal": "conformal",
        "SplitConformalRegressor": "conformal",
        "TransportBundle": "bundle",
        "TransportMLP": "mlp",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
