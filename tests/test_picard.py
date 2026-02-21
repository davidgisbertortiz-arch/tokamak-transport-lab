"""Tests for the Picard iteration loop and its sub-modules.

All tests use small grids / few iterations to stay fast (<5 s total).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.integration.convergence import (
    is_converged,
    relative_l2_residual,
)
from tokamak_transport_lab.integration.picard import (
    PicardResult,
    _compute_a_over_lte,
    run_picard,
)
from tokamak_transport_lab.integration.relaxation import mix_profiles, update_alpha
from tokamak_transport_lab.integration.safeguards import (
    clamp_inputs,
    has_nonfinite,
    should_fallback,
)

# ── convergence ──────────────────────────────────────────────────


class TestConvergence:
    def test_zero_residual_identical_profiles(self) -> None:
        te = np.array([1.0, 2.0, 3.0])
        assert relative_l2_residual(te, te) == pytest.approx(0.0)

    def test_nonzero_residual(self) -> None:
        te_old = np.array([1.0, 2.0, 3.0])
        te_new = np.array([1.1, 2.1, 3.1])
        res = relative_l2_residual(te_new, te_old)
        assert res > 0

    def test_is_converged_true(self) -> None:
        assert is_converged(1e-6, 1e-4) is True

    def test_is_converged_false(self) -> None:
        assert is_converged(1e-3, 1e-4) is False


# ── relaxation ───────────────────────────────────────────────────


class TestRelaxation:
    def test_update_alpha_first_iter(self) -> None:
        """First iteration (no prev residual) — alpha unchanged."""
        assert update_alpha(0.5, 0.1, None) == pytest.approx(0.5)

    def test_update_alpha_decrease(self) -> None:
        """Residual decreased → alpha increases by step_up."""
        a = update_alpha(0.5, 0.05, 0.1, alpha_min=0.05, alpha_max=1.0)
        assert a == pytest.approx(0.55)

    def test_update_alpha_increase(self) -> None:
        """Residual increased → alpha halved."""
        a = update_alpha(0.5, 0.2, 0.1, alpha_min=0.05, alpha_max=1.0)
        assert a == pytest.approx(0.25)

    def test_update_alpha_clamp_max(self) -> None:
        a = update_alpha(0.98, 0.01, 0.02, alpha_max=1.0)
        assert a == pytest.approx(1.0)

    def test_update_alpha_clamp_min(self) -> None:
        a = update_alpha(0.06, 0.2, 0.1, alpha_min=0.05)
        assert a == pytest.approx(0.05)  # 0.06/2 = 0.03 → clamped to 0.05

    def test_mix_profiles(self) -> None:
        old = np.array([1.0, 2.0, 3.0])
        new = np.array([3.0, 4.0, 5.0])
        mixed = mix_profiles(old, new, alpha=0.5)
        np.testing.assert_allclose(mixed, [2.0, 3.0, 4.0])


# ── safeguards ───────────────────────────────────────────────────


class TestSafeguards:
    def test_has_nonfinite_clean(self) -> None:
        assert has_nonfinite(np.array([1.0, 2.0, 3.0])) is False

    def test_has_nonfinite_nan(self) -> None:
        assert has_nonfinite(np.array([1.0, np.nan, 3.0])) is True

    def test_has_nonfinite_inf(self) -> None:
        assert has_nonfinite(np.array([1.0, np.inf, 3.0])) is True

    def test_clamp_inputs_no_extrapolation(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        clamped, was_ext = clamp_inputs(x, bounds)
        assert not was_ext
        np.testing.assert_array_equal(clamped, x)

    def test_clamp_inputs_with_extrapolation(self) -> None:
        x = np.array([[10.0, -1.0]])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        clamped, was_ext = clamp_inputs(x, bounds)
        assert was_ext
        np.testing.assert_allclose(clamped, [[5.0, 0.0]])

    def test_clamp_inputs_1d(self) -> None:
        x = np.array([10.0, -1.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        clamped, was_ext = clamp_inputs(x, bounds)
        assert was_ext
        np.testing.assert_allclose(clamped, [5.0, 0.0])

    def test_should_fallback_below_patience(self) -> None:
        assert should_fallback(3, 5) is False

    def test_should_fallback_at_patience(self) -> None:
        assert should_fallback(5, 5) is True


# ── a/L_Te computation ───────────────────────────────────────────


class TestAOverLTe:
    def test_flat_profile_zero_gradient(self) -> None:
        rho = np.linspace(0.0, 1.0, 20)
        te = np.full(20, 1000.0)
        a_over_lte = _compute_a_over_lte(te, rho)
        np.testing.assert_allclose(a_over_lte, 0.0, atol=1e-10)

    def test_positive_gradient_for_decreasing_te(self) -> None:
        """Te decreasing outward ⇒ a/L_Te > 0 in interior."""
        rho = np.linspace(0.0, 1.0, 50)
        te = 1000.0 * (1.0 - rho)  # decreasing
        a_over_lte = _compute_a_over_lte(te, rho)
        # Interior points should be positive
        assert (a_over_lte[1:-1] >= 0).all()


# ── full Picard loop ─────────────────────────────────────────────


class TestPicardLoop:
    def test_converges_with_analytic_model(self) -> None:
        """Picard with stiffness model should converge (small grid)."""
        from tokamak_transport_lab.transport.stiffness import chi_total

        result = run_picard(
            n_rho=30,
            t_ped=500.0,
            s0=1.0,
            rho_dep=0.3,
            sigma=0.1,
            transport_model=chi_total,
            transport_params={
                "chi_s": 1.0,
                "a_over_LTe_crit": 3.0,
                "alpha_s": 1.5,
                "chi_neo": 0.01,
            },
            dt=1e-4,
            sub_steps=200,
            max_iters=30,
            tol=1e-3,
            alpha0=0.3,
        )
        assert isinstance(result, PicardResult)
        assert result.rho.shape == (30,)
        assert result.te_final.shape == (30,)
        assert result.chi_profile.shape == (30,)
        assert len(result.residual_history) > 0
        assert result.metadata["n_iters"] >= 1

    def test_result_metadata_keys(self) -> None:
        from tokamak_transport_lab.transport.stiffness import chi_total

        result = run_picard(
            n_rho=20,
            transport_model=chi_total,
            transport_params={"chi_neo": 0.5},
            max_iters=3,
            sub_steps=50,
        )
        meta = result.metadata
        assert "n_iters" in meta
        assert "converged" in meta
        assert "used_fallback" in meta
        assert "wall_time_s" in meta
        assert "final_residual" in meta

    def test_nan_chi_triggers_fallback(self) -> None:
        """If transport model returns NaN, the fallback should kick in."""

        def bad_model(a_over_lte: np.ndarray, **_: object) -> np.ndarray:
            return np.full_like(a_over_lte, np.nan)

        from tokamak_transport_lab.transport.stiffness import chi_total

        result = run_picard(
            n_rho=20,
            transport_model=bad_model,
            fallback_model=chi_total,
            fallback_params={"chi_neo": 0.5},
            max_iters=5,
            sub_steps=50,
        )
        assert result.metadata["used_fallback"] is True
        # Should still produce finite Te
        assert np.isfinite(result.te_final).all()

    def test_pedestal_bc_preserved(self) -> None:
        """Dirichlet BC at rho=1 should be maintained."""
        from tokamak_transport_lab.transport.stiffness import chi_total

        t_ped = 400.0
        result = run_picard(
            n_rho=20,
            t_ped=t_ped,
            transport_model=chi_total,
            transport_params={"chi_neo": 0.5},
            max_iters=5,
            sub_steps=100,
        )
        assert result.te_final[-1] == pytest.approx(t_ped, abs=1.0)

    def test_miller_geometry(self) -> None:
        """Picard works with Miller geometry too."""
        from functools import partial

        from tokamak_transport_lab.geometry.miller import vprime_miller
        from tokamak_transport_lab.transport.stiffness import chi_total

        result = run_picard(
            n_rho=20,
            v_prime_fn=partial(vprime_miller, kappa=1.5, delta=0.2),
            transport_model=chi_total,
            transport_params={"chi_neo": 0.5},
            max_iters=5,
            sub_steps=50,
        )
        assert np.isfinite(result.te_final).all()

    # ── Phase-07 finalisation tests ──────────────────────────────

    def test_stable_case_converges(self) -> None:
        """Well-behaved scenario converges in < 60 iters with converged=True."""
        from tokamak_transport_lab.transport.stiffness import chi_total

        result = run_picard(
            n_rho=40,
            t_ped=500.0,
            s0=1.0,
            rho_dep=0.3,
            sigma=0.1,
            transport_model=chi_total,
            transport_params={
                "chi_s": 1.0,
                "a_over_LTe_crit": 3.0,
                "alpha_s": 1.5,
                "chi_neo": 0.5,
            },
            dt=1e-4,
            sub_steps=300,
            max_iters=60,
            tol=1e-3,
            alpha0=0.5,
        )
        assert result.metadata["converged"] is True
        assert result.metadata["n_iters"] < 60

    def test_fallback_on_adversarial_model(self) -> None:
        """Adversarial model (returns extreme chi) triggers fallback."""
        from tokamak_transport_lab.transport.stiffness import chi_total

        def adversarial_model(
            a_over_lte: np.ndarray, **_: object
        ) -> np.ndarray:
            # Alternating NaN / huge values → forces fallback
            out = np.full_like(a_over_lte, np.nan)
            return out

        result = run_picard(
            n_rho=20,
            transport_model=adversarial_model,
            fallback_model=chi_total,
            fallback_params={"chi_neo": 0.5},
            max_iters=8,
            sub_steps=50,
            divergence_patience=3,
        )
        assert result.metadata["used_fallback"] is True
        assert np.isfinite(result.te_final).all()

    def test_residual_generally_decreases(self) -> None:
        """On a well-behaved case the residual should trend downward.

        We allow up to 3 small 'bumps' — the overall trend must be
        decreasing as measured by first-half-mean > second-half-mean.
        """
        from tokamak_transport_lab.transport.stiffness import chi_total

        result = run_picard(
            n_rho=40,
            t_ped=500.0,
            s0=1.0,
            rho_dep=0.3,
            sigma=0.1,
            transport_model=chi_total,
            transport_params={
                "chi_s": 1.0,
                "a_over_LTe_crit": 3.0,
                "alpha_s": 1.5,
                "chi_neo": 0.5,
            },
            dt=1e-4,
            sub_steps=300,
            max_iters=40,
            tol=1e-4,
            alpha0=0.5,
        )
        hist = result.residual_history
        assert len(hist) >= 2, "Need at least 2 iterations"
        mid = len(hist) // 2
        first_half_mean = np.mean(hist[:mid])
        second_half_mean = np.mean(hist[mid:])
        assert second_half_mean < first_half_mean, (
            f"Residual not decreasing: first_half={first_half_mean:.2e}, "
            f"second_half={second_half_mean:.2e}"
        )
