"""Legacy constant-volume rescaling retained for old configuration files.

This is NOT Miller geometry: kappa*(1+delta**2/2) is constant and cancels
from the diffusion operator. It has no effect on temperatures. The supported
physical scope of the current model is circular radial diffusion.
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
    """Return the historical constant rescaling; this adds no shaping physics."""
    if not np.isfinite([kappa, delta]).all() or kappa <= 0:
        raise ValueError("Legacy geometry parameters must be finite with kappa > 0")
    rho = np.asarray(rho, dtype=np.float64)
    shape_factor = kappa * (1.0 + 0.5 * delta**2)
    return shape_factor * rho
