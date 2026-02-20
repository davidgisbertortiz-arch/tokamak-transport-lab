"""Tests for the critical-gradient stiffness transport model."""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.transport.stiffness import chi_total, chi_turbulent, normalised_flux


class TestChiTurbulent:
    """Core stiffness‐model behaviour."""

    def test_zero_below_threshold(self) -> None:
        """chi_turb must be exactly 0 when a/L_Te < a/L_Te,crit."""
        a_over_LTe = np.array([0.0, 1.0, 2.0, 2.99])
        result = chi_turbulent(a_over_LTe, chi_s=2.0, a_over_LTe_crit=3.0, alpha_s=1.5)
        np.testing.assert_array_equal(result, 0.0)

    def test_positive_above_threshold(self) -> None:
        """chi_turb must be > 0 when a/L_Te > a/L_Te,crit."""
        a_over_LTe = np.array([3.01, 5.0, 10.0])
        result = chi_turbulent(a_over_LTe, chi_s=2.0, a_over_LTe_crit=3.0, alpha_s=1.5)
        assert np.all(result > 0)

    def test_monotonicity(self) -> None:
        """Increasing a/L_Te must increase chi_turb for fixed params."""
        a_over_LTe = np.linspace(3.0, 10.0, 50)
        result = chi_turbulent(a_over_LTe, chi_s=2.0, a_over_LTe_crit=3.0, alpha_s=1.5)
        dchi = np.diff(result)
        assert np.all(dchi >= 0)

    def test_scalar_input(self) -> None:
        """Must work with a single scalar input."""
        val = chi_turbulent(5.0, chi_s=1.0, a_over_LTe_crit=3.0, alpha_s=1.0)
        assert val == pytest.approx(2.0, rel=1e-12)

    def test_at_threshold_is_zero(self) -> None:
        """Exactly at the critical gradient: chi_turb = 0."""
        val = chi_turbulent(3.0, chi_s=1.0, a_over_LTe_crit=3.0, alpha_s=1.5)
        assert val == pytest.approx(0.0, abs=1e-15)


class TestChiTotal:
    """chi_total = chi_turb + chi_neo."""

    def test_floor_below_threshold(self) -> None:
        """Below threshold, chi_total = chi_neo exactly."""
        result = chi_total(
            np.array([0.0, 1.0, 2.0]),
            chi_s=2.0,
            a_over_LTe_crit=3.0,
            alpha_s=1.5,
            chi_neo=0.01,
        )
        np.testing.assert_allclose(result, 0.01)

    def test_always_ge_chi_neo(self) -> None:
        """chi_total >= chi_neo always."""
        a_over_LTe = np.linspace(0.0, 15.0, 200)
        chi_neo = 0.01
        result = chi_total(
            a_over_LTe, chi_s=3.0, a_over_LTe_crit=4.0, alpha_s=1.0, chi_neo=chi_neo
        )
        assert np.all(result >= chi_neo - 1e-15)


class TestNormalisedFlux:
    """Qe/Q_gB = chi_total * (a/L_Te)."""

    def test_zero_gradient_gives_zero_flux(self) -> None:
        """Zero gradient → zero flux."""
        val = normalised_flux(0.0, chi_s=1.0, a_over_LTe_crit=3.0, alpha_s=1.5, chi_neo=0.01)
        assert val == pytest.approx(0.0, abs=1e-15)

    def test_positive_flux(self) -> None:
        """Above threshold, flux must be > 0."""
        val = normalised_flux(5.0, chi_s=2.0, a_over_LTe_crit=3.0, alpha_s=1.5, chi_neo=0.01)
        assert val > 0
