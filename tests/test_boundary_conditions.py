"""Regression tests for Dirichlet boundary conditions at ρ=1.

Ensures Te[-1] == Te_ped and Ti[-1] == Ti_ped after Picard convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.integration.multichannel_picard import run_picard_multichannel
from tokamak_transport_lab.integration.picard import run_picard
from tokamak_transport_lab.transport.stiffness import chi_total


class TestDirichletBC:
    """Boundary values must match pedestal temperatures exactly."""

    _BC_TOL = 1e-9

    def test_single_channel_te_boundary(self) -> None:
        """Te[-1] == t_ped after single-channel Picard."""
        t_ped = 600.0
        result = run_picard(
            n_rho=30,
            t_ped=t_ped,
            s0=2.0,
            transport_model=chi_total,
            transport_params={
                "chi_s": 1.0,
                "a_over_LTe_crit": 3.0,
                "alpha_s": 1.5,
                "chi_neo": 0.01,
            },
            max_iters=20,
            tol=1e-3,
            sub_steps=100,
        )
        assert abs(result.te_final[-1] - t_ped) < self._BC_TOL, (
            f"Te[-1]={result.te_final[-1]}, expected {t_ped}"
        )

    def test_multichannel_te_boundary(self) -> None:
        """Te[-1] == te_ped after multichannel Picard."""
        te_ped = 500.0
        result = run_picard_multichannel(
            n_rho=30,
            te_ped=te_ped,
            ti_ped=400.0,
            s0_e=1.5,
            max_iters=20,
            tol=1e-3,
            sub_steps=100,
        )
        assert abs(result.te_final[-1] - te_ped) < self._BC_TOL, (
            f"Te[-1]={result.te_final[-1]}, expected {te_ped}"
        )

    def test_multichannel_ti_boundary(self) -> None:
        """Ti[-1] == ti_ped after multichannel Picard."""
        ti_ped = 350.0
        result = run_picard_multichannel(
            n_rho=30,
            te_ped=500.0,
            ti_ped=ti_ped,
            s0_e=1.5,
            max_iters=20,
            tol=1e-3,
            sub_steps=100,
        )
        assert abs(result.ti_final[-1] - ti_ped) < self._BC_TOL, (
            f"Ti[-1]={result.ti_final[-1]}, expected {ti_ped}"
        )

    @pytest.mark.parametrize("s0_e", [0.3, 2.0, 8.0])
    def test_bc_holds_across_heating_levels(self, s0_e: float) -> None:
        """BC must hold regardless of heating power."""
        te_ped, ti_ped = 500.0, 400.0
        result = run_picard_multichannel(
            n_rho=30,
            te_ped=te_ped,
            ti_ped=ti_ped,
            s0_e=s0_e,
            max_iters=25,
            tol=1e-3,
            sub_steps=100,
        )
        assert abs(result.te_final[-1] - te_ped) < self._BC_TOL
        assert abs(result.ti_final[-1] - ti_ped) < self._BC_TOL
