"""Split conformal prediction for regression.

Provides distribution-free prediction intervals with guaranteed marginal
coverage at any user-chosen miscoverage level alpha.  Accepts both numpy
arrays and torch tensors as inputs.

Algorithm (Vovk et al., Lei et al.):
    1.  On a held-out calibration set, compute non-conformity scores
        s_i = |y_i - yhat_i|.
    2.  q_hat = quantile(s, ceil((1 - alpha)(n + 1)) / n).
    3.  Interval for a new point: [yhat - q_hat, yhat + q_hat].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import torch

    _HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_TORCH = False

ArrayLike = Union["NDArray[np.float64]", "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> NDArray[np.float64]:
    """Convert numpy array *or* torch Tensor to ``float64`` ndarray."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


class SplitConformal:
    """Split conformal wrapper for symmetric absolute-residual scores.

    Supports both numpy arrays and torch tensors as inputs.

    Parameters
    ----------
    q_hat : float or None
        Pre-computed conformal quantile.  Set by :meth:`fit` or loaded
        from disk.
    alpha : float
        Miscoverage level used during fit.
    """

    def __init__(self) -> None:
        self.q_hat: float | None = None
        self.alpha: float | None = None

    # ── calibration ──────────────────────────────────────────────

    def fit(
        self,
        y_cal: ArrayLike,
        yhat_cal: ArrayLike,
        alpha: float = 0.1,
    ) -> SplitConformal:
        """Calibrate from a held-out calibration set.

        Parameters
        ----------
        y_cal : array-like (n,)
            True target values (numpy array or torch Tensor).
        yhat_cal : array-like (n,)
            Point predictions on calibration set.
        alpha : float
            Desired miscoverage level (e.g. 0.1 for 90 % coverage).

        Returns
        -------
        self
        """
        y_np = _to_numpy(y_cal)
        yhat_np = _to_numpy(yhat_cal)
        scores = np.abs(y_np - yhat_np)
        n = len(scores)
        # Finite-sample-valid quantile level
        level = np.ceil((1 - alpha) * (n + 1)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self.alpha = alpha
        return self

    # ── prediction ───────────────────────────────────────────────

    def interval(
        self,
        yhat: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return prediction interval ``(lower, upper)``.

        Parameters
        ----------
        yhat : array-like
            Point predictions for new data (numpy or torch Tensor).

        Returns
        -------
        lower, upper : numpy arrays (same shape as *yhat*)
        """
        if self.q_hat is None:
            raise RuntimeError("Call .fit() before .interval().")
        yhat_np = _to_numpy(yhat)
        return yhat_np - self.q_hat, yhat_np + self.q_hat

    # Alias requested by spec
    predict_interval = interval

    # ── persistence ──────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save calibration state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"q_hat": self.q_hat, "alpha": self.alpha}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SplitConformal:
        """Load calibration state from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        obj.q_hat = data["q_hat"]
        obj.alpha = data["alpha"]
        return obj


# Alias for spec compatibility
SplitConformalRegressor = SplitConformal
