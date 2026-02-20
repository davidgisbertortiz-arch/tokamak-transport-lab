"""Boundary-condition helpers for the 1-D solver.

Provides functions to enforce:
- Neumann (symmetry) at ρ = 0:  ∂T/∂ρ = 0
- Dirichlet (pedestal) at ρ = 1: T = T_ped
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def enforce_dirichlet(
    T: NDArray[np.float64],
    t_ped: float,
) -> None:
    """Set T at the last grid point to the pedestal value (in-place)."""
    T[-1] = t_ped


def check_symmetry(
    T: NDArray[np.float64],
    rho: NDArray[np.float64],
    *,
    rtol: float = 0.05,
) -> bool:
    """Return True if the gradient at ρ=0 is small relative to the max gradient.

    Parameters
    ----------
    T : array (N,)
        Temperature profile.
    rho : array (N,)
        Radial grid.
    rtol : float
        Maximum allowed ratio |dT/dρ(0)| / max|dT/dρ|.
    """
    drho = rho[1] - rho[0]
    grad_0 = abs(T[1] - T[0]) / drho
    grad_max = float(np.max(np.abs(np.diff(T)))) / drho
    if grad_max == 0.0:
        return True
    return bool(grad_0 / grad_max < rtol)
