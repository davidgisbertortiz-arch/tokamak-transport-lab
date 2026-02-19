"""Semi-analytic reference tests."""

from __future__ import annotations

import numpy as np

from tokamak_transport_lab.solver.analytic import steady_state_reference


class TestAnalyticReference:
    """Verify the quadrature-based steady-state computation."""

    def test_boundary_value(self) -> None:
        """T(1) must equal T_ped."""
        rho = np.linspace(0.0, 1.0, 200)
        T = steady_state_reference(rho, chi0=1.0, s0=1.0, rho_dep=0.3, sigma=0.1, t_ped=500.0)
        assert abs(T[-1] - 500.0) < 1e-6

    def test_monotonic_decrease(self) -> None:
        """For a central source, T should decrease from core to edge."""
        rho = np.linspace(0.0, 1.0, 200)
        T = steady_state_reference(rho, chi0=1.0, s0=1.0, rho_dep=0.0, sigma=0.2, t_ped=100.0)
        # Allow tiny non-monotonicity from numerics at the boundary
        dT = np.diff(T)
        assert np.all(dT[1:] <= 1e-10), "T should be monotonically non-increasing"

    def test_higher_source_gives_higher_temp(self) -> None:
        """Doubling the source should raise the core temperature."""
        rho = np.linspace(0.0, 1.0, 200)
        T1 = steady_state_reference(rho, chi0=1.0, s0=1.0, rho_dep=0.3, sigma=0.1, t_ped=500.0)
        T2 = steady_state_reference(rho, chi0=1.0, s0=2.0, rho_dep=0.3, sigma=0.1, t_ped=500.0)
        assert T2[0] > T1[0]
