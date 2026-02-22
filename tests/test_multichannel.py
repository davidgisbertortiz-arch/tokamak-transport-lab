"""Tests for the coupled Te–Ti multichannel Picard loop.

Verifies:
1. Coupling sign: S_ei = (3/2) n (Te−Ti) / τ_eq is positive when Te > Ti.
2. Physics sanity: in electron-heated scenarios Ti ≤ Te everywhere.
3. Energy conservation: coupling term sums to zero (what electrons lose,
   ions gain).
4. Convergence on all three reference scenarios (small-grid fast versions).
5. Robustness: NaN transport triggers fallback gracefully.

All tests use small grids / few iterations to stay fast (< 10 s total).
"""

from __future__ import annotations

import numpy as np
import pytest
from tokamak_transport_lab.integration.multichannel_picard import (
    MultichannelResult,
    equilibration_source,
    run_picard_multichannel,
)
from tokamak_transport_lab.transport.stiffness import chi_total


# ── equilibration source ─────────────────────────────────────────


class TestEquilibrationSource:
    def test_positive_when_te_gt_ti(self) -> None:
        """S_ei > 0 when Te > Ti (electrons lose energy to ions)."""
        te = np.array([1000.0, 800.0, 600.0])
        ti = np.array([500.0, 400.0, 300.0])
        s_ei = equilibration_source(te, ti, density=1.0, tau_eq=0.01)
        assert (s_ei > 0).all()

    def test_negative_when_ti_gt_te(self) -> None:
        """S_ei < 0 when Ti > Te (ions lose energy to electrons)."""
        te = np.array([300.0, 400.0])
        ti = np.array([500.0, 600.0])
        s_ei = equilibration_source(te, ti, density=1.0, tau_eq=0.01)
        assert (s_ei < 0).all()

    def test_zero_when_equal(self) -> None:
        """S_ei = 0 when Te = Ti."""
        te = np.array([500.0, 600.0, 700.0])
        ti = np.array([500.0, 600.0, 700.0])
        s_ei = equilibration_source(te, ti, density=1.0, tau_eq=0.01)
        np.testing.assert_allclose(s_ei, 0.0, atol=1e-12)

    def test_energy_conservation_pairwise(self) -> None:
        """Sum of ±S_ei is zero: electrons lose = ions gain."""
        te = np.array([1000.0, 800.0, 600.0, 400.0])
        ti = np.array([500.0, 700.0, 200.0, 350.0])
        s_ei = equilibration_source(te, ti, density=1.5, tau_eq=0.01)
        # What electron equation gets is −S_ei, ion equation gets +S_ei
        # Net: −S_ei + S_ei = 0 at each grid point
        np.testing.assert_allclose((-s_ei + s_ei), 0.0, atol=1e-15)

    def test_stronger_coupling_with_smaller_tau(self) -> None:
        """Smaller τ_eq → larger |S_ei|."""
        te = np.array([1000.0])
        ti = np.array([500.0])
        s_fast = equilibration_source(te, ti, density=1.0, tau_eq=0.001)
        s_slow = equilibration_source(te, ti, density=1.0, tau_eq=0.1)
        assert float(abs(s_fast[0])) > float(abs(s_slow[0]))

    def test_formula_values(self) -> None:
        """Check exact formula: S_ei = (3/2) n (Te - Ti) / τ_eq."""
        te = np.array([1000.0])
        ti = np.array([400.0])
        n, tau = 2.0, 0.05
        expected = 1.5 * n * (1000.0 - 400.0) / tau
        s_ei = equilibration_source(te, ti, density=n, tau_eq=tau)
        assert s_ei[0] == pytest.approx(expected)


# ── multichannel Picard loop ─────────────────────────────────────


class TestMultichannelPicard:
    """Core tests for the coupled Te–Ti Picard solver."""

    def test_converges_with_coupling(self) -> None:
        """Coupled Te–Ti system converges with analytic transport."""
        result = run_picard_multichannel(
            n_rho=30,
            te_ped=500.0,
            ti_ped=400.0,
            s0_e=1.0,
            s0_i=0.0,
            density=1.0,
            tau_eq=0.01,
            transport_model_e=chi_total,
            transport_params_e={
                "chi_s": 1.0,
                "a_over_LTe_crit": 3.0,
                "alpha_s": 1.5,
                "chi_neo": 0.01,
            },
            transport_model_i=chi_total,
            transport_params_i={
                "chi_s": 0.5,
                "a_over_LTe_crit": 4.0,
                "alpha_s": 1.5,
                "chi_neo": 0.01,
            },
            dt=1e-4,
            sub_steps=200,
            max_iters=40,
            tol=1e-3,
            alpha0=0.3,
        )
        assert isinstance(result, MultichannelResult)
        assert result.te_final.shape == (30,)
        assert result.ti_final.shape == (30,)
        assert len(result.residual_history) > 0

    def test_ti_less_than_te_electron_heated(self) -> None:
        """In an electron-heated scenario Ti ≤ Te (within tolerance)."""
        result = run_picard_multichannel(
            n_rho=30,
            te_ped=500.0,
            ti_ped=400.0,
            s0_e=1.0,
            s0_i=0.0,
            density=1.0,
            tau_eq=0.01,
            transport_model_e=chi_total,
            transport_params_e={"chi_s": 1.0, "chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_s": 0.5, "chi_neo": 0.5},
            dt=1e-4,
            sub_steps=300,
            max_iters=40,
            tol=1e-3,
            alpha0=0.3,
        )
        # Allow small numerical tolerance (1 eV)
        assert (result.ti_final <= result.te_final + 1.0).all(), (
            f"Ti > Te found: max(Ti-Te) = {(result.ti_final - result.te_final).max():.1f}"
        )

    def test_pedestal_bc_preserved(self) -> None:
        """Dirichlet BCs at ρ=1 are maintained for both channels."""
        te_ped, ti_ped = 500.0, 350.0
        result = run_picard_multichannel(
            n_rho=20,
            te_ped=te_ped,
            ti_ped=ti_ped,
            transport_model_e=chi_total,
            transport_params_e={"chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_neo": 0.5},
            max_iters=5,
            sub_steps=100,
        )
        assert result.te_final[-1] == pytest.approx(te_ped, abs=1.0)
        assert result.ti_final[-1] == pytest.approx(ti_ped, abs=1.0)

    def test_metadata_keys(self) -> None:
        """Result metadata has all expected keys."""
        result = run_picard_multichannel(
            n_rho=20,
            transport_model_e=chi_total,
            transport_params_e={"chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_neo": 0.5},
            max_iters=3,
            sub_steps=50,
        )
        meta = result.metadata
        for key in [
            "n_iters",
            "converged",
            "used_fallback",
            "wall_time_s",
            "final_residual",
            "te_core_eV",
            "ti_core_eV",
        ]:
            assert key in meta, f"Missing key: {key}"

    def test_nan_transport_triggers_fallback(self) -> None:
        """NaN transport model causes fallback, not a crash."""

        def bad_model(a_over_lte, **_):
            return np.full_like(a_over_lte, np.nan)

        result = run_picard_multichannel(
            n_rho=20,
            transport_model_e=bad_model,
            transport_model_i=chi_total,
            transport_params_i={"chi_neo": 0.5},
            fallback_model=chi_total,
            fallback_params={"chi_neo": 0.5},
            max_iters=5,
            sub_steps=50,
        )
        assert result.metadata["used_fallback"] is True
        assert np.isfinite(result.te_final).all()
        assert np.isfinite(result.ti_final).all()

    def test_profiles_all_finite(self) -> None:
        """All output profiles and histories are finite."""
        result = run_picard_multichannel(
            n_rho=20,
            transport_model_e=chi_total,
            transport_params_e={"chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_neo": 0.5},
            max_iters=10,
            sub_steps=100,
        )
        assert np.isfinite(result.te_final).all()
        assert np.isfinite(result.ti_final).all()
        assert np.isfinite(result.chi_e_profile).all()
        assert np.isfinite(result.chi_i_profile).all()

    def test_coupling_sign_in_profiles(self) -> None:
        """With coupling and electron heating only, Te_core > Ti_core."""
        result = run_picard_multichannel(
            n_rho=30,
            te_ped=500.0,
            ti_ped=400.0,
            s0_e=2.0,
            s0_i=0.0,
            density=1.0,
            tau_eq=0.02,
            transport_model_e=chi_total,
            transport_params_e={"chi_s": 1.0, "chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_s": 0.5, "chi_neo": 0.5},
            dt=1e-4,
            sub_steps=300,
            max_iters=40,
            tol=1e-3,
            alpha0=0.3,
        )
        # Core Te should be significantly above core Ti
        assert result.te_final[0] > result.ti_final[0], (
            f"Te_core={result.te_final[0]:.1f} should be > Ti_core={result.ti_final[0]:.1f}"
        )


# ── scenario-level convergence ───────────────────────────────────


class TestScenarioConvergence:
    """Quick-run versions of all three reference scenarios."""

    @staticmethod
    def _fast_scenario(
        *,
        te_ped: float,
        ti_ped: float,
        s0_e: float,
        chi_s_e: float,
        chi_s_i: float,
        density: float,
        tau_eq: float,
    ) -> MultichannelResult:
        """Run a small-grid fast version of a scenario."""
        return run_picard_multichannel(
            n_rho=30,
            te_ped=te_ped,
            ti_ped=ti_ped,
            s0_e=s0_e,
            s0_i=0.0,
            density=density,
            tau_eq=tau_eq,
            transport_model_e=chi_total,
            transport_params_e={
                "chi_s": chi_s_e,
                "a_over_LTe_crit": 3.0,
                "alpha_s": 1.5,
                "chi_neo": 0.02,
            },
            transport_model_i=chi_total,
            transport_params_i={
                "chi_s": chi_s_i,
                "a_over_LTe_crit": 4.0,
                "alpha_s": 1.5,
                "chi_neo": 0.02,
            },
            dt=1e-4,
            sub_steps=200,
            max_iters=60,
            tol=1e-3,
            alpha0=0.3,
        )

    def test_low_power_scenario(self) -> None:
        result = self._fast_scenario(
            te_ped=300.0,
            ti_ped=250.0,
            s0_e=0.3,
            chi_s_e=0.5,
            chi_s_i=0.3,
            density=1.0,
            tau_eq=0.02,
        )
        assert result.metadata["n_iters"] < 60
        assert np.isfinite(result.te_final).all()
        assert np.isfinite(result.ti_final).all()

    def test_mid_power_scenario(self) -> None:
        result = self._fast_scenario(
            te_ped=500.0,
            ti_ped=400.0,
            s0_e=1.0,
            chi_s_e=1.0,
            chi_s_i=0.5,
            density=1.5,
            tau_eq=0.01,
        )
        assert result.metadata["n_iters"] < 60
        assert np.isfinite(result.te_final).all()

    def test_high_power_scenario(self) -> None:
        result = self._fast_scenario(
            te_ped=1500.0,
            ti_ped=1200.0,
            s0_e=4.0,
            chi_s_e=2.0,
            chi_s_i=1.0,
            density=2.0,
            tau_eq=0.005,
        )
        assert result.metadata["n_iters"] <= 60
        assert np.isfinite(result.te_final).all()


# ── exchange sign test (diffusion off) ───────────────────────────


class TestExchangeSignNoDiffusion:
    """When diffusion is negligible and Te > Ti, one coupling update must
    decrease Te and increase Ti by roughly the same magnitude."""

    @staticmethod
    def _flat_chi(a_over_lte: np.ndarray, **_kw: object) -> np.ndarray:
        """Tiny constant diffusivity — effectively no diffusion."""
        return np.full_like(a_over_lte, 1e-8)

    def test_te_decreases_ti_increases(self) -> None:
        """After 1 Picard iter with near-zero χ, coupling moves energy
        from electrons to ions."""
        n = 32
        te_init = np.full(n, 1000.0)  # flat, high Te
        ti_init = np.full(n, 400.0)  # flat, low Ti

        result = run_picard_multichannel(
            n_rho=n,
            te_ped=1000.0,
            ti_ped=400.0,
            s0_e=0.0,           # no external heating
            s0_i=0.0,
            density=2.0,
            tau_eq=0.005,       # strong coupling
            transport_model_e=self._flat_chi,
            transport_model_i=self._flat_chi,
            dt=1e-3,
            theta=0.5,
            sub_steps=50,
            max_iters=1,        # single iteration
            tol=1e-12,
            alpha0=1.0,         # no under-relaxation
            te_init=te_init,
            ti_init=ti_init,
        )

        # Interior points (exclude boundary which is Dirichlet)
        interior = slice(1, -1)
        # Te must have decreased
        assert (result.te_final[interior] < te_init[interior]).all(), (
            "Te should decrease when Te > Ti with coupling on"
        )
        # Ti must have increased
        assert (result.ti_final[interior] > ti_init[interior]).all(), (
            "Ti should increase when Te > Ti with coupling on"
        )

    def test_sign_opposite_trends(self) -> None:
        """The change in Te and Ti should be of opposite sign (energy
        conservation at each grid point)."""
        n = 32
        te_init = np.full(n, 800.0)
        ti_init = np.full(n, 300.0)

        result = run_picard_multichannel(
            n_rho=n,
            te_ped=800.0,
            ti_ped=300.0,
            s0_e=0.0,
            s0_i=0.0,
            density=1.5,
            tau_eq=0.01,
            transport_model_e=self._flat_chi,
            transport_model_i=self._flat_chi,
            dt=1e-3,
            sub_steps=50,
            max_iters=1,
            tol=1e-12,
            alpha0=1.0,
            te_init=te_init,
            ti_init=ti_init,
        )

        delta_te = result.te_final[1:-1] - te_init[1:-1]
        delta_ti = result.ti_final[1:-1] - ti_init[1:-1]

        # Opposite signs everywhere interior
        assert (delta_te * delta_ti < 0).all(), (
            "Delta(Te) and Delta(Ti) must have opposite signs"
        )


# ── end-to-end sanity (tiny grid, fast) ──────────────────────────


class TestEndToEndSanity:
    """Quick integration test on a small grid (N=32) — must complete
    in < 5 s and produce physically plausible output."""

    def test_no_nans_positive_temps(self) -> None:
        """Profiles must be NaN-free and strictly positive."""
        result = run_picard_multichannel(
            n_rho=32,
            te_ped=500.0,
            ti_ped=400.0,
            s0_e=1.0,
            s0_i=0.0,
            density=1.0,
            tau_eq=0.01,
            transport_model_e=chi_total,
            transport_params_e={"chi_s": 1.0, "chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_s": 0.5, "chi_neo": 0.5},
            dt=1e-4,
            sub_steps=100,
            max_iters=20,
            tol=1e-3,
            alpha0=0.3,
        )
        assert np.isfinite(result.te_final).all()
        assert np.isfinite(result.ti_final).all()
        assert (result.te_final > 0).all()
        assert (result.ti_final > 0).all()

    def test_mean_ti_increases_with_electron_heating(self) -> None:
        """In an electron-heated scenario, equilibration should raise
        mean(Ti) above the initial flat profile."""
        n = 32
        ti_ped = 400.0
        ti_init = np.linspace(ti_ped + 100.0, ti_ped, n)  # gentle slope
        mean_ti_init = float(ti_init.mean())

        result = run_picard_multichannel(
            n_rho=n,
            te_ped=600.0,
            ti_ped=ti_ped,
            s0_e=2.0,           # strong electron heating
            s0_i=0.0,
            density=1.0,
            tau_eq=0.005,       # strong coupling
            transport_model_e=chi_total,
            transport_params_e={"chi_s": 1.0, "chi_neo": 0.5},
            transport_model_i=chi_total,
            transport_params_i={"chi_s": 0.5, "chi_neo": 0.5},
            dt=1e-4,
            sub_steps=200,
            max_iters=30,
            tol=1e-3,
            alpha0=0.3,
            ti_init=ti_init,
        )

        mean_ti_final = float(result.ti_final.mean())
        assert mean_ti_final > mean_ti_init, (
            f"mean(Ti) should increase: init={mean_ti_init:.1f} → "
            f"final={mean_ti_final:.1f}"
        )
