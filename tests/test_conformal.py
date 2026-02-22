"""Tests for SplitConformal prediction intervals."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tokamak_transport_lab.surrogate.conformal import SplitConformal


class TestSplitConformal:
    """Verify conformal interval properties."""

    @pytest.fixture()
    def calibrated(self) -> SplitConformal:
        """A fitted SplitConformal on synthetic data."""
        rng = np.random.default_rng(0)
        y_cal = rng.normal(0, 1, size=200)
        yhat_cal = y_cal + rng.normal(0, 0.1, size=200)
        sc = SplitConformal()
        sc.fit(y_cal, yhat_cal, alpha=0.1)
        return sc

    def test_fit_sets_q_hat(self, calibrated: SplitConformal) -> None:
        assert calibrated.q_hat is not None
        assert calibrated.q_hat > 0
        assert calibrated.alpha == 0.1

    def test_interval_shapes(self, calibrated: SplitConformal) -> None:
        yhat = np.linspace(-2, 2, 50)
        lo, hi = calibrated.interval(yhat)
        assert lo.shape == yhat.shape
        assert hi.shape == yhat.shape

    def test_interval_contains_center(self, calibrated: SplitConformal) -> None:
        """Center of interval should equal the point prediction."""
        yhat = np.array([0.0, 1.0, -1.0])
        lo, hi = calibrated.interval(yhat)
        midpoint = 0.5 * (lo + hi)
        np.testing.assert_allclose(midpoint, yhat)

    def test_coverage_on_calibration_data(self) -> None:
        """Coverage on an i.i.d. test set should be near 1 - alpha."""
        rng = np.random.default_rng(42)
        n = 5000
        y = rng.normal(0, 1, size=n)
        yhat = y + rng.normal(0, 0.2, size=n)

        # Calibrate on first half, test on second half
        sc = SplitConformal()
        sc.fit(y[: n // 2], yhat[: n // 2], alpha=0.1)
        lo, hi = sc.interval(yhat[n // 2 :])
        cov = np.mean((y[n // 2 :] >= lo) & (y[n // 2 :] <= hi))
        assert cov >= 0.85, f"coverage = {cov:.3f}, expected >= 0.85"

    def test_save_load_roundtrip(self, calibrated: SplitConformal) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conformal.json"
            calibrated.save(path)

            loaded = SplitConformal.load(path)

        assert loaded.q_hat == pytest.approx(calibrated.q_hat)
        assert loaded.alpha == calibrated.alpha

    def test_interval_before_fit_raises(self) -> None:
        sc = SplitConformal()
        with pytest.raises(RuntimeError, match="fit"):
            sc.interval(np.array([1.0]))
