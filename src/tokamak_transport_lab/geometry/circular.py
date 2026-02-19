"""Circular flux-surface geometry: V'(ρ) = ρ."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def v_prime(rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return V'(ρ) = ρ for circular geometry."""
    return np.copy(rho)
