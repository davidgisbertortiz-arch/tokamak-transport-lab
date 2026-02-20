"""Regression metrics for surrogate evaluation.

All functions accept plain numpy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def r2_score(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Coefficient of determination (R²).

    Returns 1.0 for a perfect prediction, 0.0 for a constant-mean
    prediction, and negative for worse-than-mean.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def mae(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def max_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(y_true - y_pred)))
