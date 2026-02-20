"""Smoke test: package imports and version."""

from __future__ import annotations


def test_version() -> None:
    import tokamak_transport_lab

    assert tokamak_transport_lab.__version__ == "0.1.0"


def test_subpackage_imports() -> None:
    """All sub-packages should be importable."""
    import importlib

    subpackages = [
        "tokamak_transport_lab.solver",
        "tokamak_transport_lab.physics",
        "tokamak_transport_lab.transport",
        "tokamak_transport_lab.geometry",
        "tokamak_transport_lab.surrogate",
        "tokamak_transport_lab.data",
        "tokamak_transport_lab.integration",
        "tokamak_transport_lab.evaluation",
        "tokamak_transport_lab.visualization",
        "tokamak_transport_lab.utils",
    ]
    for pkg in subpackages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"
