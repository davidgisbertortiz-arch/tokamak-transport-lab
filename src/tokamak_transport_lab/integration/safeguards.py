"""Safeguards: NaN/Inf detection, input clamping, fallback logic.

Provides utility functions used by the Picard loop to ensure robustness:
- Detect non-finite values in chi or Te profiles.
- Clamp surrogate inputs to training-data ranges and flag extrapolation.
- Decide when to fall back from the surrogate to the analytic model.

Limitations
-----------
- Input-range clamping is per-feature independent (no joint OOD detection).
- Fallback is binary: once triggered by patience exhaustion it is permanent
  for the remaining iterations of that Picard run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def has_nonfinite(arr: NDArray[np.float64]) -> bool:
    """Return True if *arr* contains any NaN or Inf."""
    return bool(~np.isfinite(arr).all())


def clamp_inputs(
    inputs: NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> tuple[NDArray[np.float64], bool]:
    """Clamp each column of *inputs* to the given [lo, hi] bounds.

    Parameters
    ----------
    inputs : array (N, D) or (D,)
        Feature matrix (one row per grid point or sample).
    bounds : list of (lo, hi) pairs
        One pair per feature column.

    Returns
    -------
    clamped : array (same shape)
        Clamped copy of *inputs*.
    was_extrapolating : bool
        True if any value was outside bounds before clamping.
    """
    clamped = np.array(inputs, dtype=np.float64, copy=True)
    was_extrapolating = False
    if clamped.ndim == 1:
        for j, (lo, hi) in enumerate(bounds):
            if clamped[j] < lo or clamped[j] > hi:
                was_extrapolating = True
            clamped[j] = np.clip(clamped[j], lo, hi)
    else:
        for j, (lo, hi) in enumerate(bounds):
            col = clamped[:, j]
            if float(col.min()) < lo or float(col.max()) > hi:
                was_extrapolating = True
            clamped[:, j] = np.clip(col, lo, hi)
    return clamped, was_extrapolating


def should_fallback(
    divergence_counter: int,
    patience: int,
) -> bool:
    """Return True when we should fall back to the analytic model.

    Parameters
    ----------
    divergence_counter : int
        Number of consecutive iterations where the residual increased.
    patience : int
        Maximum allowed consecutive increases before triggering fallback.
    """
    return divergence_counter >= patience
