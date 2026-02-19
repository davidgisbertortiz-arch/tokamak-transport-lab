"""Tests for physics.sources module."""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.physics.sources import gaussian_source


class TestGaussianSource:
    """Verify properties of the Gaussian heating source."""

    @pytest.fixture()
    def rho(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, 500)

    def test_peak_at_rho_dep(self, rho: np.ndarray) -> None:
        """Maximum should be near the deposition radius."""
        S = gaussian_source(rho, s0=2.0, rho_dep=0.3, sigma=0.1)
        peak_idx = int(np.argmax(S))
        assert abs(rho[peak_idx] - 0.3) < (rho[1] - rho[0])

    def test_peak_amplitude(self, rho: np.ndarray) -> None:
        """Peak value should equal s0 when rho_dep is on the grid."""
        S = gaussian_source(rho, s0=5.0, rho_dep=0.0, sigma=0.1)
        assert S[0] == pytest.approx(5.0, rel=1e-12)

    def test_positivity(self, rho: np.ndarray) -> None:
        """Gaussian source must be strictly positive."""
        S = gaussian_source(rho, s0=1.0, rho_dep=0.5, sigma=0.05)
        assert np.all(S > 0)

    def test_shape(self, rho: np.ndarray) -> None:
        """Output shape should match input grid."""
        S = gaussian_source(rho, s0=1.0, rho_dep=0.3, sigma=0.1)
        assert S.shape == rho.shape

    def test_zero_amplitude(self, rho: np.ndarray) -> None:
        """s0 = 0 should give an all-zero source."""
        S = gaussian_source(rho, s0=0.0, rho_dep=0.3, sigma=0.1)
        np.testing.assert_array_equal(S, 0.0)
