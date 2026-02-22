"""Regression and UQ metrics for surrogate evaluation.

All functions accept plain numpy arrays.

Limitations
-----------
- ``regression_ece_gaussian`` assumes a Gaussian predictive distribution;
  for ensemble predictions the mean and std of the member outputs are
  treated as the parameters of a single Gaussian.
- ``coverage`` checks marginal (not conditional) coverage.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr  # Gaussian CDF

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── point-prediction metrics ─────────────────────────────────────


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


# ── interval / UQ metrics ───────────────────────────────────────


def coverage(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Empirical coverage: fraction of y_true within [lower, upper]."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def regression_ece_gaussian(
    y_true: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error for a Gaussian predictive distribution.

    For each sample, compute the PIT value u_i = Phi((y_i - mu_i) / sigma_i).
    If calibrated, u ~ Uniform(0, 1).  Bin the u values and compare the
    observed frequency in each bin to the expected frequency (1 / n_bins).

    Parameters
    ----------
    y_true : array (n,)
    mu : array (n,)
        Predicted mean.
    sigma : array (n,)
        Predicted standard deviation (> 0).
    n_bins : int
        Number of equi-spaced bins on [0, 1].

    Returns
    -------
    ece : float
        Weighted absolute calibration error in [0, 1].
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    u = ndtr((np.asarray(y_true) - np.asarray(mu)) / sigma)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(u)
    for lo, hi in itertools.pairwise(bin_edges):
        mask = (u >= lo) & (u < hi)
        observed = float(mask.sum()) / n
        expected = 1.0 / n_bins
        ece += abs(observed - expected)
    return ece / 2.0  # normalise so perfect calibration gives 0


def reliability_bins_gaussian(
    y_true: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    n_bins: int = 15,
) -> dict[str, NDArray[np.float64]]:
    """Return binned PIT frequencies for a reliability diagram.

    Returns
    -------
    data : dict
        ``"bin_centres"`` — (n_bins,) midpoints of [0,1] bins.
        ``"observed"``    — (n_bins,) observed frequency in each bin.
        ``"expected"``    — (n_bins,) expected frequency (uniform = 1/n_bins).
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    u = ndtr((np.asarray(y_true) - np.asarray(mu)) / sigma)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    observed = np.zeros(n_bins)
    n = len(u)
    for k, (lo, hi) in enumerate(itertools.pairwise(bin_edges)):
        observed[k] = float(((u >= lo) & (u < hi)).sum()) / n

    expected = np.full(n_bins, 1.0 / n_bins)
    return {"bin_centres": centres, "observed": observed, "expected": expected}


# ── interval-based UQ metrics ───────────────────────────────────


def ece_regression_interval(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for prediction intervals.

    Bins samples by interval width and compares per-bin empirical
    coverage to the overall average coverage.

    Parameters
    ----------
    y_true : array (n,)
    lower, upper : array (n,)
        Prediction interval bounds.
    n_bins : int
        Number of width-quantile bins.

    Returns
    -------
    ece : float
        Weighted absolute calibration error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    widths = upper - lower
    contained = (y_true >= lower) & (y_true <= upper)
    overall_cov = float(contained.mean())

    n = len(y_true)
    bin_edges = np.quantile(widths, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[-1] += 1e-10  # include rightmost point

    ece = 0.0
    for lo_edge, hi_edge in itertools.pairwise(bin_edges):
        mask = (widths >= lo_edge) & (widths < hi_edge)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_cov = float(contained[mask].mean())
        ece += (count / n) * abs(bin_cov - overall_cov)
    return ece


def reliability_bins_interval(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Reliability diagram data for prediction intervals.

    Bins samples by interval width and returns per-bin empirical coverage
    and mean interval width.

    Returns
    -------
    data : dict
        ``"bin_centres"``   — (n_bins,) midpoints of width-based bins.
        ``"observed_cov"``  — (n_bins,) per-bin empirical coverage.
        ``"mean_width"``    — (n_bins,) per-bin mean interval width.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    widths = upper - lower
    contained = (y_true >= lower) & (y_true <= upper)
    bin_edges = np.quantile(widths, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[-1] += 1e-10

    centres = np.zeros(n_bins)
    obs_cov = np.zeros(n_bins)
    mean_w = np.zeros(n_bins)

    for k, (lo_edge, hi_edge) in enumerate(itertools.pairwise(bin_edges)):
        mask = (widths >= lo_edge) & (widths < hi_edge)
        centres[k] = 0.5 * (lo_edge + hi_edge)
        count = int(mask.sum())
        if count > 0:
            obs_cov[k] = float(contained[mask].mean())
            mean_w[k] = float(widths[mask].mean())

    return {"bin_centres": centres, "observed_cov": obs_cov, "mean_width": mean_w}
