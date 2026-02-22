"""Tests for the two-channel CN solver (solver/multichannel.py).

Covers:
1. Exchange step energy conservation (Te+Ti conserved).
2. Exchange step decay sign / magnitude.
3. Single-channel consistency: with tau_eq → ∞ and s0_i=0, Te should
   match the single-channel solver output.
4. Coupled solve: Te > Ti in electron-heated scenario.
5. Dirichlet BCs preserved for both channels.
6. Residual history is recorded and trends downward.
7. Custom source / v_prime / init arrays accepted.
"""

from __future__ import annotations

import numpy as np
import pytest
from tokamak_transport_lab.solver.multichannel import (
    _exchange_step,
    solve_two_channel,
)


# ── exchange step ────────────────────────────────────────────────


class TestExchangeStep:
    def test_sum_conserved(self) -> None:
        """Te + Ti is conserved through the exchange step."""
        te = np.array([1000.0, 800.0, 600.0])
        ti = np.array([400.0, 500.0, 300.0])
        te_new, ti_new = _exchange_step(te, ti, dt=1e-4, density=1.0, tau_eq=0.01)
        np.testing.assert_allclose(te_new + ti_new, te + ti, rtol=1e-12)

    def test_difference_decays(self) -> None:
        """Te − Ti decays (|D_new| < |D_old|) for positive dt."""
        te = np.array([1000.0, 800.0])
        ti = np.array([400.0, 500.0])
        te_new, ti_new = _exchange_step(te, ti, dt=1e-4, density=1.0, tau_eq=0.01)
        d_old = te - ti
        d_new = te_new - ti_new
        assert np.all(np.abs(d_new) <= np.abs(d_old) + 1e-15)

    def test_no_exchange_at_equilibrium(self) -> None:
        """When Te = Ti there is no change."""
        t = np.array([500.0, 600.0, 700.0])
        te_new, ti_new = _exchange_step(t, t.copy(), dt=1e-4, density=1.0, tau_eq=0.01)
        np.testing.assert_allclose(te_new, t, atol=1e-12)
        np.testing.assert_allclose(ti_new, t, atol=1e-12)

    def test_stronger_decay_smaller_tau(self) -> None:
        """Smaller τ_eq → faster exchange → smaller remaining difference."""
        te = np.array([1000.0])
        ti = np.array([400.0])
        _, ti_fast = _exchange_step(te, ti, dt=1e-3, density=1.0, tau_eq=0.001)
        _, ti_slow = _exchange_step(te, ti, dt=1e-3, density=1.0, tau_eq=1.0)
        # Fast coupling makes Ti closer to Te
        assert float(ti_fast[0]) > float(ti_slow[0])

    def test_exact_formula(self) -> None:
        """Verify against analytic formula: D(dt) = D(0) * exp(-2 ν dt)."""
        te = np.array([1000.0])
        ti = np.array([400.0])
        n, tau, dt = 2.0, 0.05, 1e-3
        nu_ei = 1.5 * n / tau
        expected_decay = np.exp(-2.0 * nu_ei * dt)
        d0 = 600.0
        te_new, ti_new = _exchange_step(te, ti, dt=dt, density=n, tau_eq=tau)
        d_new = float(te_new[0] - ti_new[0])
        assert d_new == pytest.approx(d0 * expected_decay, rel=1e-12)


# ── two-channel solve ────────────────────────────────────────────


class TestSolveTwoChannel:
    def test_output_keys(self) -> None:
        """Output dict has all required keys."""
        out = solve_two_channel(n_rho=20, n_steps=10)
        for key in ["rho", "Te", "Ti", "chi_e", "chi_i", "source_e", "source_i",
                     "residual_history"]:
            assert key in out, f"Missing key: {key}"

    def test_output_shapes(self) -> None:
        n = 30
        out = solve_two_channel(n_rho=n, n_steps=50)
        assert out["rho"].shape == (n,)
        assert out["Te"].shape == (n,)
        assert out["Ti"].shape == (n,)
        assert out["chi_e"].shape == (n,)
        assert out["chi_i"].shape == (n,)
        assert out["source_e"].shape == (n,)
        assert out["source_i"].shape == (n,)
        assert out["residual_history"].shape == (50,)

    def test_dirichlet_bcs(self) -> None:
        """Pedestal BCs are enforced at ρ = 1."""
        te_ped, ti_ped = 500.0, 350.0
        out = solve_two_channel(
            n_rho=30, te_ped=te_ped, ti_ped=ti_ped, n_steps=100
        )
        assert out["Te"][-1] == pytest.approx(te_ped, abs=0.1)
        assert out["Ti"][-1] == pytest.approx(ti_ped, abs=0.1)

    def test_te_gt_ti_electron_heated(self) -> None:
        """Electron heating only (s0_i=0): Te ≥ Ti everywhere."""
        out = solve_two_channel(
            n_rho=50,
            chi_e=1.0,
            chi_i=0.5,
            s0_e=2.0,
            s0_i=0.0,
            te_ped=500.0,
            ti_ped=400.0,
            density=1.0,
            tau_eq=0.01,
            n_steps=3000,
        )
        assert (out["Ti"] <= out["Te"] + 1.0).all(), (
            f"Ti > Te: max diff = {(out['Ti'] - out['Te']).max():.1f}"
        )

    def test_all_finite(self) -> None:
        out = solve_two_channel(n_rho=30, n_steps=200)
        for key in ["Te", "Ti", "chi_e", "chi_i"]:
            assert np.isfinite(out[key]).all(), f"Non-finite in {key}"

    def test_residual_decreases_overall(self) -> None:
        """Residual should generally decrease toward steady state."""
        out = solve_two_channel(
            n_rho=40, chi_e=1.0, chi_i=0.5, n_steps=2000, dt=1e-4
        )
        hist = out["residual_history"]
        assert len(hist) == 2000
        # Compare first quarter vs last quarter
        q = len(hist) // 4
        assert np.mean(hist[-q:]) < np.mean(hist[:q])

    def test_custom_source_arrays(self) -> None:
        """Accepts pre-computed source arrays."""
        n = 30
        src = np.ones(n) * 0.5
        out = solve_two_channel(
            n_rho=n, source_e=src, source_i=src * 0.3, n_steps=50
        )
        np.testing.assert_array_equal(out["source_e"], src)
        np.testing.assert_allclose(out["source_i"], src * 0.3)

    def test_custom_vprime(self) -> None:
        """Accepts a custom V'(ρ) array (e.g. Miller-like)."""
        n = 30
        rho = np.linspace(0.0, 1.0, n)
        vp = 1.5 * rho  # elongated circular
        out = solve_two_channel(n_rho=n, v_prime=vp, n_steps=100)
        assert np.isfinite(out["Te"]).all()
        assert np.isfinite(out["Ti"]).all()

    def test_custom_init_profiles(self) -> None:
        """Accepts custom initial profiles."""
        n = 30
        te0 = np.linspace(2000.0, 500.0, n)
        ti0 = np.linspace(1500.0, 400.0, n)
        out = solve_two_channel(
            n_rho=n, te_init=te0, ti_init=ti0, n_steps=50
        )
        assert out["Te"].shape == (n,)
        assert out["Ti"].shape == (n,)

    def test_single_channel_consistency(self) -> None:
        """With no coupling (huge τ_eq) and s0_i=0, Te matches single-channel.

        Ion channel gets no heating and huge τ_eq means negligible exchange,
        so the electron channel evolves independently.
        """
        from tokamak_transport_lab.solver.crank_nicolson import solve as solve_single

        kw = dict(n_rho=50, dt=1e-4, n_steps=2000, theta=0.5)
        single = solve_single(chi0=1.0, s0=1.0, t_ped=500.0, **kw)

        two_ch = solve_two_channel(
            chi_e=1.0,
            chi_i=1.0,
            s0_e=1.0,
            s0_i=0.0,
            te_ped=500.0,
            ti_ped=500.0,
            density=1.0,
            tau_eq=1e10,  # effectively no exchange
            **kw,
        )
        # Te from two-channel should match single-channel within ~1%
        np.testing.assert_allclose(
            two_ch["Te"], single["Te"], rtol=0.01,
            err_msg="Two-channel Te differs from single-channel beyond 1%",
        )

    def test_spatially_varying_chi(self) -> None:
        """Accepts spatially varying chi arrays."""
        n = 40
        rho = np.linspace(0.0, 1.0, n)
        chi_e = 0.5 + rho  # increasing outward
        chi_i = np.full(n, 0.3)
        out = solve_two_channel(
            n_rho=n, chi_e=chi_e, chi_i=chi_i, n_steps=200
        )
        assert np.isfinite(out["Te"]).all()
        assert np.isfinite(out["Ti"]).all()
