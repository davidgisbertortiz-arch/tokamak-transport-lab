"""Miller-parameterised flux-surface geometry.

Provides an effective volume-element derivative $V'(\\rho)$ that depends on
elongation $\\kappa$ and triangularity $\\delta$.  In the limit
$\\kappa = 1,\\ \\delta = 0$ the result recovers the circular expression
$V'(\\rho) = \\rho$.

This is a *minimal* Miller model sufficient to exercise the pipeline.
The full Miller parameterisation involves solving a Grad-Shafranov-like
system for the metric coefficients; here we use the lowest-order analytic
approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def vprime_miller(
    rho: NDArray[np.float64],
    kappa: float = 1.0,
    delta: float = 0.0,
) -> NDArray[np.float64]:
    r"""Effective volume-element derivative for shaped flux surfaces.

    Uses the lowest-order approximation:

    .. math::
        V'(\rho) \approx \kappa\,(1 + 0.5\,\delta^2)\;\rho

    This gives the correct circular limit ($\kappa=1, \delta=0 \Rightarrow
    V'= \rho$) and captures the leading shaping effects: elongation scales
    the cross-section area while triangularity enters at second order.

    Parameters
    ----------
    rho : array (N,)
        Normalised radial coordinate.
    kappa : float
        Elongation ($\kappa \ge 1$).
    delta : float
        Triangularity ($0 \le \delta \le 0.5$ typical).

    Returns
    -------
    vp : array (N,)
        $V'(\rho)$ in the same units as the input grid.
    """
    rho = np.asarray(rho, dtype=np.float64)
    shape_factor = kappa * (1.0 + 0.5 * delta**2)
    return shape_factor * rho
