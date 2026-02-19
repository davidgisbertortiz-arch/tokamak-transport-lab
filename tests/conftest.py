"""Shared pytest fixtures for tokamak-transport-lab."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def default_solver_kwargs() -> dict:
    """Return keyword arguments for a small, fast solver run."""
    return {
        "n_rho": 50,
        "chi0": 1.0,
        "s0": 1.0,
        "rho_dep": 0.3,
        "sigma": 0.1,
        "t_ped": 500.0,
        "dt": 1e-4,
        "n_steps": 3000,
        "theta": 0.5,
    }


@pytest.fixture()
def rho_grid() -> np.ndarray:
    """A small uniform grid on [0, 1]."""
    return np.linspace(0.0, 1.0, 50)
