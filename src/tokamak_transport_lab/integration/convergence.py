"""Convergence monitoring for the Picard iteration loop.

Tracks the relative L2 residual between successive temperature profiles
and decides when the iteration has converged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def relative_l2_residual(
    te_new: NDArray[np.float64],
    te_old: NDArray[np.float64],
) -> float:
    """Compute ||Te_{k+1} - Te_k||_2 / ||Te_k||_2.

    Parameters
    ----------
    te_new, te_old : array (N,)
        Temperature profiles at consecutive Picard iterations.

    Returns
    -------
    residual : float
        Non-negative relative residual.  Returns 0.0 if ``te_old`` is
        identically zero (degenerate case).
    """
    norm_old = float(np.linalg.norm(te_old))
    if norm_old < 1e-30:
        return 0.0
    return float(np.linalg.norm(te_new - te_old)) / norm_old


def is_converged(residual: float, tol: float) -> bool:
    """Return True when the residual drops below tolerance."""
    return residual < tol
