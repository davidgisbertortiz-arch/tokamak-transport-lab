"""Geometry module tests."""

from __future__ import annotations

import numpy as np

from tokamak_transport_lab.geometry.circular import v_prime
from tokamak_transport_lab.geometry.miller import vprime_miller


def test_v_prime_circular() -> None:
    """V'(ρ) = ρ for circular geometry."""
    rho = np.linspace(0.0, 1.0, 50)
    vp = v_prime(rho)
    np.testing.assert_array_almost_equal(vp, rho)


# ── Miller geometry ──────────────────────────────────────────────


class TestMillerVprime:
    """vprime_miller contract checks."""

    def test_circular_limit_matches(self) -> None:
        """kappa=1, delta=0 must recover circular V'(rho) = rho."""
        rho = np.linspace(0.0, 1.0, 50)
        vp_miller = vprime_miller(rho, kappa=1.0, delta=0.0)
        vp_circ = v_prime(rho)
        np.testing.assert_allclose(vp_miller, vp_circ, atol=1e-14)

    def test_elongation_changes_result(self) -> None:
        """kappa != 1 should give different (larger) V' than circular."""
        rho = np.linspace(0.0, 1.0, 50)
        vp_circ = vprime_miller(rho, kappa=1.0, delta=0.0)
        vp_elong = vprime_miller(rho, kappa=1.6, delta=0.0)
        # skip rho=0 where both are zero
        assert np.all(vp_elong[1:] > vp_circ[1:])

    def test_triangularity_changes_result(self) -> None:
        """delta != 0 should give different (larger) V' than delta=0."""
        rho = np.linspace(0.0, 1.0, 50)
        vp_base = vprime_miller(rho, kappa=1.3, delta=0.0)
        vp_tri = vprime_miller(rho, kappa=1.3, delta=0.4)
        assert np.all(vp_tri[1:] > vp_base[1:])
