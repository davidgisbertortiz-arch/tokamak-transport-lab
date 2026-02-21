"""Lowest-order Miller parameterization of flux-surface geometry.

Provides a V'(rho) function that captures the leading-order effects of
elongation (kappa) and triangularity (delta) on the flux-surface volume
element.  At kappa=1, delta=0 this recovers the circular limit V'=rho.

Limitations
-----------
- This is a *lowest-order* approximation.  Higher-order Shafranov-shift,
  squareness, and up-down asymmetry corrections are neglected.
- The formula assumes concentric flux surfaces with no radial dependence
  of the shaping parameters (i.e. kappa and delta are constant in rho).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def vprime_miller(
    rho: NDArray[np.float64],
    *,
    kappa: float = 1.0,
    delta: float = 0.0,
) -> NDArray[np.float64]:
    r"""Return the volume-element derivative V'(rho) with Miller shaping.

    .. math::

        V'(\rho) = \rho \, \kappa \, (1 + \tfrac{1}{2}\,\delta^{2})

    Parameters
    ----------
    rho : array
        Normalised radial coordinate, rho in [0, 1].
    kappa : float
        Elongation (>=1).  kappa=1 is circular.
    delta : float
        Triangularity.  delta=0 is symmetric.

    Returns
    -------
    vp : array (same shape as *rho*)
        V'(rho).
    """
    return np.asarray(rho, dtype=np.float64) * kappa * (1.0 + 0.5 * delta**2)
