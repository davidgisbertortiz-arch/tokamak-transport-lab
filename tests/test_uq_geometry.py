"""Tests for Miller geometry and UQ metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.evaluation.metrics import (
    coverage,
    regression_ece_gaussian,
    reliability_bins_gaussian,
)
from tokamak_transport_lab.geometry.circular import v_prime
from tokamak_transport_lab.geometry.miller import vprime_miller

# ── Miller geometry ──────────────────────────────────────────────


class TestMillerGeometry:
    """Verify Miller V'(rho) behaviour."""

    @pytest.fixture()
    def rho(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, 100)

    def test_circular_limit(self, rho: np.ndarray) -> None:
        """kappa=1, delta=0 must recover circular V' = rho."""
        vp_miller = vprime_miller(rho, kappa=1.0, delta=0.0)
        vp_circ = v_prime(rho)
        np.testing.assert_allclose(vp_miller, vp_circ, atol=1e-14)

    def test_elongation_scales_up(self, rho: np.ndarray) -> None:
        """Higher kappa should give larger V' everywhere (except rho=0)."""
        vp1 = vprime_miller(rho, kappa=1.0, delta=0.0)
        vp2 = vprime_miller(rho, kappa=1.7, delta=0.0)
        # skip rho=0 where both are 0
        assert np.all(vp2[1:] > vp1[1:])

    def test_triangularity_effect(self, rho: np.ndarray) -> None:
        """Non-zero delta should increase V' relative to delta=0."""
        vp0 = vprime_miller(rho, kappa=1.5, delta=0.0)
        vp1 = vprime_miller(rho, kappa=1.5, delta=0.4)
        assert np.all(vp1[1:] > vp0[1:])

    def test_zero_at_origin(self, rho: np.ndarray) -> None:
        """V'(0) = 0 regardless of shaping."""
        assert vprime_miller(rho, kappa=1.8, delta=0.3)[0] == 0.0


# ── UQ metrics ───────────────────────────────────────────────────


class TestCoverage:
    """Empirical interval coverage."""

    def test_perfect_coverage(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        assert coverage(y, lo, hi) == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        y = np.array([10.0, 20.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage(y, lo, hi) == pytest.approx(0.0)

    def test_partial_coverage(self) -> None:
        y = np.array([1.0, 5.0, 3.0, 10.0])
        lo = np.array([0.0, 0.0, 2.0, 0.0])
        hi = np.array([2.0, 4.0, 4.0, 5.0])
        # 1.0 in [0,2] yes; 5.0 in [0,4] no; 3.0 in [2,4] yes; 10 in [0,5] no
        assert coverage(y, lo, hi) == pytest.approx(0.5)


class TestRegressionECE:
    """Expected Calibration Error for Gaussian predictions."""

    def test_perfect_calibration_low_ece(self) -> None:
        """Samples drawn from a known Gaussian should have low ECE."""
        rng = np.random.default_rng(99)
        n = 10_000
        mu = rng.uniform(-5, 5, n)
        sigma = rng.uniform(0.5, 2.0, n)
        y = rng.normal(mu, sigma)
        ece = regression_ece_gaussian(y, mu, sigma, n_bins=15)
        assert ece < 0.05, f"ECE = {ece:.4f}, expected < 0.05"

    def test_reliability_bins_shape(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(size=100)
        mu = np.zeros(100)
        sigma = np.ones(100)
        data = reliability_bins_gaussian(y, mu, sigma, n_bins=10)
        assert data["bin_centres"].shape == (10,)
        assert data["observed"].shape == (10,)
        assert data["expected"].shape == (10,)
