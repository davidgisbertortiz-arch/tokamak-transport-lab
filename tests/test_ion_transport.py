"""Tests for ion transport models (transport/ion_transport.py).

Covers:
1. ScaledQeTransport: chi_i = qi_factor × chi_e.
2. chi_total_ion: independent stiffness model for ions.
3. make_ion_transport factory: mode dispatch, invalid mode error.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_transport_lab.transport.ion_transport import (
    ScaledQeTransport,
    chi_total_ion,
    make_ion_transport,
)
from tokamak_transport_lab.transport.stiffness import chi_total

# ── ScaledQeTransport ────────────────────────────────────────────


class TestScaledQeTransport:
    def test_scaled_matches_factor(self) -> None:
        """chi_i = qi_factor * chi_e for the same gradient."""
        a = np.linspace(0.0, 10.0, 50)
        params = {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}
        factor = 0.7
        model = ScaledQeTransport(chi_total, electron_params=params, qi_factor=factor)

        chi_i = model(a)
        chi_e = chi_total(a, **params)
        np.testing.assert_allclose(chi_i, factor * chi_e, rtol=1e-12)

    def test_factor_one_equals_electron(self) -> None:
        """qi_factor=1.0 → chi_i == chi_e."""
        a = np.array([0.0, 3.0, 5.0, 8.0])
        model = ScaledQeTransport(chi_total, qi_factor=1.0)
        chi_i = model(a)
        chi_e = chi_total(a)
        np.testing.assert_allclose(chi_i, chi_e, rtol=1e-12)

    def test_uses_electron_gradient_when_provided(self) -> None:
        """When _a_over_lte is provided, chi_e is evaluated on that."""
        a_lte = np.array([5.0, 6.0])
        a_lti = np.array([1.0, 1.0])  # below threshold → would give chi_neo only
        params = {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}
        model = ScaledQeTransport(chi_total, electron_params=params, qi_factor=1.0)

        # With _a_over_lte: uses electron gradient (above threshold)
        chi_with = model(a_lti, _a_over_lte=a_lte)
        chi_e_ref = chi_total(a_lte, **params)
        np.testing.assert_allclose(chi_with, chi_e_ref, rtol=1e-12)

    def test_falls_back_to_ion_gradient(self) -> None:
        """Without _a_over_lte, uses a_over_lti as proxy."""
        a_lti = np.array([5.0, 6.0])
        params = {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}
        model = ScaledQeTransport(chi_total, electron_params=params, qi_factor=0.5)

        chi_i = model(a_lti)
        chi_e_ref = chi_total(a_lti, **params)
        np.testing.assert_allclose(chi_i, 0.5 * chi_e_ref, rtol=1e-12)

    def test_output_shape_matches_input(self) -> None:
        model = ScaledQeTransport(chi_total, qi_factor=0.8)
        a = np.linspace(0.0, 10.0, 30)
        result = model(a)
        assert result.shape == a.shape

    def test_always_positive(self) -> None:
        """chi_i should always be > 0 (neoclassical floor)."""
        model = ScaledQeTransport(chi_total, electron_params={"chi_neo": 0.01}, qi_factor=0.5)
        a = np.array([0.0, 1.0, 2.0])  # all below threshold
        chi_i = model(a)
        assert (chi_i > 0).all()


# ── chi_total_ion (stiffness) ────────────────────────────────────


class TestChiTotalIon:
    def test_zero_below_threshold(self) -> None:
        """Turbulent component is zero below critical gradient."""
        a = np.array([0.0, 1.0, 3.9])  # default crit = 4.0
        chi = chi_total_ion(a)
        # Should be neoclassical floor only
        expected = np.full_like(a, 0.01)
        np.testing.assert_allclose(chi, expected, rtol=1e-10)

    def test_nonzero_above_threshold(self) -> None:
        a = np.array([5.0, 8.0, 10.0])
        chi = chi_total_ion(a, chi_s=0.5, a_over_LTe_crit=4.0)
        assert (chi > 0.01).all()

    def test_matches_electron_model_same_params(self) -> None:
        """With same parameters, ion model == electron model."""
        a = np.linspace(0.0, 12.0, 50)
        params = {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}
        chi_ion = chi_total_ion(a, **params)
        chi_elec = chi_total(a, **params)
        np.testing.assert_allclose(chi_ion, chi_elec, rtol=1e-12)

    def test_custom_ion_params(self) -> None:
        """Ion model respects different parameters from electron."""
        a = np.array([5.0])
        chi_default = chi_total_ion(a)  # chi_s=0.5, crit=4.0
        chi_stiffer = chi_total_ion(a, chi_s=2.0, a_over_LTe_crit=2.0)
        assert float(chi_stiffer[0]) > float(chi_default[0])


# ── make_ion_transport factory ───────────────────────────────────


class TestMakeIonTransport:
    def test_stiffness_mode(self) -> None:
        ion_params = {"chi_s": 0.5, "a_over_LTe_crit": 4.0, "chi_neo": 0.02}
        model, params = make_ion_transport(mode="stiffness", ion_params=ion_params)
        a = np.array([5.0, 6.0])
        result = model(a, **params)
        expected = chi_total_ion(a, **ion_params)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_scaled_qe_mode(self) -> None:
        e_params = {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}
        model, params = make_ion_transport(
            mode="scaled_qe",
            electron_model=chi_total,
            electron_params=e_params,
            qi_factor=0.6,
        )
        a = np.array([5.0, 8.0])
        chi_i = model(a, **params)
        chi_e = chi_total(a, **e_params)
        np.testing.assert_allclose(chi_i, 0.6 * chi_e, rtol=1e-12)

    def test_default_mode_is_stiffness(self) -> None:
        model, _params = make_ion_transport()
        assert model is chi_total_ion

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ti_transport_mode"):
            make_ion_transport(mode="invalid_mode")

    def test_scaled_qe_default_electron_model(self) -> None:
        """scaled_qe mode without electron_model falls back to chi_total."""
        model, params = make_ion_transport(mode="scaled_qe", qi_factor=0.8)
        a = np.array([5.0])
        result = model(a, **params)
        assert result.shape == (1,)
        assert float(result[0]) > 0
