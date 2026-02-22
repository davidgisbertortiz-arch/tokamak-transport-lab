"""Smoke tests for the Crank–Nicolson solver (Week 1 MVP).

Tests
-----
1. BC enforcement: T(ρ=1) == T_ped exactly.
2. Symmetry BC: dT/dρ|_{ρ=0} ≈ 0 (finite-difference check).
3. Steady-state matches semi-analytic reference with L² error < 1e-3.
4. Residual history is monotonically decreasing (after transient).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.solver.analytic import steady_state_reference
from tokamak_transport_lab.solver.crank_nicolson import solve


class TestBoundaryConditions:
    """Verify that boundary conditions are enforced."""

    def test_dirichlet_at_pedestal(self, default_solver_kwargs: dict) -> None:
        """T at ρ=1 must equal T_ped exactly."""
        result = solve(**default_solver_kwargs)
        assert result["Te"][-1] == pytest.approx(default_solver_kwargs["t_ped"], abs=1e-12)

    def test_symmetry_at_origin(self, default_solver_kwargs: dict) -> None:
        """dT/dρ at ρ=0 should be ≈ 0 (ghost-point approach)."""
        result = solve(**default_solver_kwargs)
        # Forward difference: (T[1] - T[0]) / Δρ
        drho = result["rho"][1] - result["rho"][0]
        grad_0 = (result["Te"][1] - result["Te"][0]) / drho
        # Should be small relative to peak gradient
        grad_mid = np.max(np.abs(np.diff(result["Te"]))) / drho
        assert abs(grad_0) < 0.05 * grad_mid


class TestSteadyStateAccuracy:
    """Compare solver output with the semi-analytic steady-state reference."""

    @pytest.fixture()
    def _run_converged(self) -> dict:
        """Run solver long enough to approach steady state."""
        return solve(
            n_rho=100,
            chi0=1.0,
            s0=1.0,
            rho_dep=0.3,
            sigma=0.1,
            t_ped=500.0,
            dt=1e-3,
            n_steps=20_000,
            theta=0.5,
        )

    def test_l2_error_below_threshold(self, _run_converged: dict) -> None:
        """L² relative error < 1e-3 compared to quadrature reference."""
        res = _run_converged
        T_ref = steady_state_reference(
            res["rho"], chi0=1.0, s0=1.0, rho_dep=0.3, sigma=0.1, t_ped=500.0
        )
        l2_err = float(np.linalg.norm(res["Te"] - T_ref) / np.linalg.norm(T_ref))
        assert l2_err < 1e-3, f"L² relative error = {l2_err:.6e}"


class TestResidualDecay:
    """Check that the solver converges in time."""

    def test_residual_decreases(self, default_solver_kwargs: dict) -> None:
        """The trailing portion of the residual history should be decreasing."""
        result = solve(**default_solver_kwargs)
        resid = result["residual_history"]
        # After initial transient (first 10 %), check trend
        tail = resid[len(resid) // 10 :]
        # Allow a few bumps — check overall trend via polyfit slope
        x = np.arange(len(tail), dtype=float)
        slope = np.polyfit(x, np.log10(tail + 1e-30), 1)[0]
        assert slope < 0, f"Residual slope = {slope:.4f}, expected negative"


class TestSmokeEndToEnd:
    """Minimal end-to-end: solver runs without error and returns expected keys."""

    def test_output_keys(self, default_solver_kwargs: dict) -> None:
        result = solve(**default_solver_kwargs)
        for key in ("rho", "Te", "chi", "source", "residual_history"):
            assert key in result, f"Missing key: {key}"

    def test_output_shapes(self, default_solver_kwargs: dict) -> None:
        result = solve(**default_solver_kwargs)
        n = default_solver_kwargs["n_rho"]
        assert result["rho"].shape == (n,)
        assert result["Te"].shape == (n,)

    def test_temperature_positive(self, default_solver_kwargs: dict) -> None:
        result = solve(**default_solver_kwargs)
        assert np.all(result["Te"] > 0), "Temperature must be positive everywhere"
