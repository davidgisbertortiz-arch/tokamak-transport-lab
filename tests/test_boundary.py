"""Tests for solver.boundary module."""

from __future__ import annotations

import numpy as np

from tokamak_transport_lab.solver.boundary import check_symmetry, enforce_dirichlet


class TestEnforceDirichlet:
    """Verify Dirichlet BC enforcement."""

    def test_sets_last_value(self) -> None:
        T = np.array([1000.0, 800.0, 600.0, 400.0])
        enforce_dirichlet(T, t_ped=200.0)
        assert T[-1] == 200.0

    def test_does_not_alter_interior(self) -> None:
        T = np.array([1000.0, 800.0, 600.0, 400.0])
        enforce_dirichlet(T, t_ped=200.0)
        np.testing.assert_array_equal(T[:3], [1000.0, 800.0, 600.0])


class TestCheckSymmetry:
    """Verify Neumann-symmetry check at ρ = 0."""

    def test_flat_profile_is_symmetric(self) -> None:
        rho = np.linspace(0.0, 1.0, 50)
        T = np.ones(50) * 500.0
        assert check_symmetry(T, rho) is True

    def test_symmetric_profile_passes(self) -> None:
        """A parabolic profile (zero gradient at 0) should pass."""
        rho = np.linspace(0.0, 1.0, 100)
        T = 1000.0 * (1.0 - rho**2)  # dT/drho = -2000ρ → 0 at ρ=0
        assert check_symmetry(T, rho) is True

    def test_asymmetric_profile_fails(self) -> None:
        """A profile with a large gradient at ρ = 0 should fail."""
        rho = np.linspace(0.0, 1.0, 100)
        T = 1000.0 * (1.0 - rho)  # dT/drho = -1000 everywhere
        # grad_0 / grad_max ≈ 1.0, which exceeds default rtol=0.05
        assert check_symmetry(T, rho) is False
