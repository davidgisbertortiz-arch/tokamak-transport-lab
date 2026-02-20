"""Heating-source profiles for the 1-D transport equation.

All functions return source arrays on a given ρ grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gaussian_source(
    rho: NDArray[np.float64],
    s0: float = 1.0,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
) -> NDArray[np.float64]:
    """Gaussian heating source centred at *rho_dep*.

    .. math::
        S(\\rho) = S_0 \\exp\\!\\left(-\\frac{(\\rho - \\rho_{\\text{dep}})^2}
                                          {2\\sigma^2}\\right)

    Parameters
    ----------
    rho : array (N,)
        Normalised radius grid.
    s0 : float
        Peak source amplitude.
    rho_dep : float
        Deposition centre in normalised radius.
    sigma : float
        Gaussian width.

    Returns
    -------
    S : array (N,)
        Source profile on *rho*.
    """
    return s0 * np.exp(-((rho - rho_dep) ** 2) / (2.0 * sigma**2))
