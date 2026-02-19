"""Geometry module tests."""

from __future__ import annotations

import numpy as np

from tokamak_transport_lab.geometry.circular import v_prime


def test_v_prime_circular() -> None:
    """V'(ρ) = ρ for circular geometry."""
    rho = np.linspace(0.0, 1.0, 50)
    vp = v_prime(rho)
    np.testing.assert_array_almost_equal(vp, rho)
