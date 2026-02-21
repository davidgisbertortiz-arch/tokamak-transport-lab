"""Tests for transport adapters and parity benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.transport.adapters import (
    SurrogateAdapter,
    SurrogateWithUQAdapter,
    analytic_adapter,
)

# ── analytic adapter ─────────────────────────────────────────────


class TestAnalyticAdapter:
    def test_matches_chi_total(self) -> None:
        """analytic_adapter must produce the same result as chi_total."""
        from tokamak_transport_lab.transport.stiffness import chi_total

        a = np.linspace(0.0, 10.0, 50)
        expected = chi_total(a, chi_s=1.0, a_over_LTe_crit=3.0, alpha_s=1.5, chi_neo=0.01)
        got = analytic_adapter(a, chi_s=1.0, a_over_LTe_crit=3.0, alpha_s=1.5, chi_neo=0.01)
        np.testing.assert_allclose(got, expected)

    def test_default_kwargs(self) -> None:
        a = np.array([5.0])
        chi = analytic_adapter(a)
        assert chi.shape == (1,)
        assert chi[0] > 0.0


# ── surrogate adapter ───────────────────────────────────────────


class _FakeMLP:
    """Minimal torch-like model returning Q_norm = 0.5 * a/L_Te."""

    def __call__(self, x):  
        import torch

        # First column is a/L_Te; return Q = 0.5 * a_over_lte
        return (0.5 * x[:, 0:1]).float()


class TestSurrogateAdapter:
    def test_returns_chi(self) -> None:
        """chi = Q_norm / a_over_lte ≈ 0.5 everywhere."""
        adapter = SurrogateAdapter(model=_FakeMLP())
        a = np.array([2.0, 4.0, 6.0])
        chi = adapter(a)
        assert chi.shape == (3,)
        np.testing.assert_allclose(chi, 0.5, atol=1e-5)

    def test_chi_floor(self) -> None:
        """At a/L_Te ≈ 0, chi should be clamped to chi_floor."""
        adapter = SurrogateAdapter(model=_FakeMLP(), chi_floor=1e-3)
        chi = adapter(np.array([0.0]))
        assert chi[0] >= 1e-3

    def test_extra_features_shape(self) -> None:
        """Extra features expand the input dimension correctly."""

        class _WiderMLP:
            def __call__(self, x):  
                import torch

                # Expect 3 columns: a_over_lte + 2 extras
                assert x.shape[1] == 3
                return torch.ones(x.shape[0], 1)

        adapter = SurrogateAdapter(
            model=_WiderMLP(),
            extra_features={"q": 1.4, "s_hat": 0.8},
        )
        chi = adapter(np.array([5.0, 10.0]))
        assert chi.shape == (2,)

    def test_picard_compatible(self) -> None:
        """Adapter works as run_picard transport_model."""
        from tokamak_transport_lab.integration.picard import run_picard

        adapter = SurrogateAdapter(model=_FakeMLP())
        result = run_picard(
            n_rho=20,
            transport_model=adapter,
            max_iters=3,
            sub_steps=50,
        )
        assert np.isfinite(result.te_final).all()


# ── surrogate + UQ adapter ──────────────────────────────────────


class TestSurrogateWithUQAdapter:
    def test_stores_bands(self) -> None:
        adapter = SurrogateWithUQAdapter(model=_FakeMLP(), q_hat=0.1)
        a = np.array([3.0, 6.0])
        chi = adapter(a)
        assert adapter.last_lower is not None
        assert adapter.last_upper is not None
        assert adapter.last_lower.shape == (2,)
        # lower ≤ point estimate ≤ upper
        np.testing.assert_array_less(adapter.last_lower - 1e-10, chi)
        np.testing.assert_array_less(chi - 1e-10, adapter.last_upper)

    def test_zero_q_hat_tight_bands(self) -> None:
        """q_hat = 0 → lower == upper == point estimate."""
        adapter = SurrogateWithUQAdapter(model=_FakeMLP(), q_hat=0.0)
        a = np.array([5.0])
        chi = adapter(a)
        np.testing.assert_allclose(adapter.last_lower, chi, atol=1e-10)
        np.testing.assert_allclose(adapter.last_upper, chi, atol=1e-10)


# ── parity benchmark ────────────────────────────────────────────


class TestParityReport:
    def test_identical_model_zero_error(self) -> None:
        """If surrogate == analytic, max_rel_error ≈ 0."""
        from tokamak_transport_lab.evaluation.benchmarks import parity_report

        report = parity_report(
            surrogate_model=analytic_adapter,
            surrogate_params={"chi_neo": 0.01},
            analytic_params={"chi_neo": 0.01},
            picard_kwargs={"n_rho": 20, "max_iters": 10, "sub_steps": 100},
        )
        assert report["max_rel_error"] < 1e-10
        assert "rmse" in report
        assert "analytic_converged" in report
        assert "surrogate_converged" in report
        assert "analytic_wall_s" in report

    def test_different_params_nonzero_error(self) -> None:
        """Mismatched chi_neo should produce nonzero discrepancy."""
        from tokamak_transport_lab.evaluation.benchmarks import parity_report

        report = parity_report(
            surrogate_model=analytic_adapter,
            surrogate_params={"chi_neo": 5.0},
            analytic_params={"chi_neo": 0.01},
            picard_kwargs={"n_rho": 20, "max_iters": 10, "sub_steps": 100},
        )
        assert report["max_rel_error"] > 0.01

    def test_save_report(self, tmp_path) -> None:
        from tokamak_transport_lab.evaluation.benchmarks import save_report

        import json

        report = {"max_rel_error": 0.01, "rmse": 0.5}
        out = tmp_path / "report.json"
        save_report(report, out)
        loaded = json.loads(out.read_text())
        assert loaded["max_rel_error"] == pytest.approx(0.01)
