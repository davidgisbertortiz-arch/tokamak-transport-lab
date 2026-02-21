"""Split conformal prediction for regression.

Provides distribution-free prediction intervals with guaranteed marginal
coverage at any user-chosen miscoverage level alpha.

Algorithm (Vovk et al., Lei et al.):
    1.  On a held-out calibration set, compute non-conformity scores
        s_i = |y_i - yhat_i|.
    2.  q_hat = quantile(s, ceil((1 - alpha)(n + 1)) / n).
    3.  Interval for a new point: [yhat - q_hat, yhat + q_hat].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplitConformal:
    """Split conformal wrapper for symmetric absolute-residual scores.

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
        y_cal: NDArray[np.float64],
        yhat_cal: NDArray[np.float64],
        alpha: float = 0.1,
    ) -> SplitConformal:
        """Calibrate from a held-out calibration set.

        Parameters
        ----------
        y_cal : array (n,)
            True target values.
        yhat_cal : array (n,)
            Point predictions on calibration set.
        alpha : float
            Desired miscoverage level (e.g. 0.1 for 90 % coverage).

        Returns
        -------
        self
        """
        scores = np.abs(y_cal - yhat_cal)
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
        yhat: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return prediction interval ``(lower, upper)``.

        Parameters
        ----------
        yhat : array
            Point predictions for new data.

        Returns
        -------
        lower, upper : arrays (same shape as *yhat*)
        """
        if self.q_hat is None:
            raise RuntimeError("Call .fit() before .interval().")
        yhat = np.asarray(yhat, dtype=np.float64)
        return yhat - self.q_hat, yhat + self.q_hat

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
