"""Regression tests for the scientific failures found in the repository audit."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from tokamak_transport_lab.integration.config import run_config as run_one_scenario
from tokamak_transport_lab.integration.multichannel_picard import run_picard_multichannel
from tokamak_transport_lab.integration.picard import run_picard
from tokamak_transport_lab.solver.analytic import steady_state_reference
from tokamak_transport_lab.solver.crank_nicolson import solve
from tokamak_transport_lab.solver.multichannel import solve_two_channel
from tokamak_transport_lab.solver.steady import stationary_solve


def test_default_transient_reaches_steady_state():
    result = solve()
    assert result["converged"]
    assert result["steady_residual"] < 1e-6


def test_tiny_time_step_is_not_steady():
    result = solve(n_steps=2, dt=1e-10)
    assert result["residual_history"][-1] < 1e-5
    assert not result["converged"]
    assert result["steady_residual"] > 1


def test_stationary_independent_of_initial_condition_and_legacy_time_knobs():
    n = 40
    outputs = []
    for start, dt in [(500, 1e-12), (1500, 1e-4), (10000, 0.1)]:
        result = run_picard(n_rho=n, te_init=np.full(n, start), dt=dt, sub_steps=1, tol=1e-7)
        assert result.metadata["converged"]
        outputs.append(result.te_final)
    for output in outputs[1:]:
        np.testing.assert_allclose(output, outputs[0], atol=1e-6, rtol=0)


def test_circular_uniform_source_exact_quadratic():
    rho = np.linspace(0, 1, 41)
    actual = stationary_solve(rho, [np.full(41, 0.4)], rho, [np.full(41, 10.0)], [100.0])[0]
    expected = 100 + 10 / (4 * 0.4) * (1 - rho * rho)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_spatial_second_order_against_independent_quadrature():
    errors = []
    for n in (31, 61, 121):
        rho = np.linspace(0, 1, n)
        src = np.exp(-((rho - 0.3) ** 2) / 0.02)
        t = stationary_solve(rho, [np.ones(n)], rho, [src], [0.0])[0]
        ref = steady_state_reference(rho, 1.0, 1.0, 0.3, 0.1, 0.0, n_quad=30001)
        errors.append(np.linalg.norm(t - ref) / np.linalg.norm(ref))
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


@pytest.mark.parametrize("name", ["low_power", "mid_power", "high_power"])
def test_actual_yaml_scenarios_converge_without_clipping(name):
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/scenarios" / f"{name}.yaml").read_text()
    )
    result = run_one_scenario(cfg)
    assert result.metadata["converged"], result.metadata
    assert result.metadata["final_residual"] <= cfg["picard"]["tol"]
    assert result.metadata["temperature_clips"] == 0
    assert not result.metadata["used_fallback"]
    assert np.all(result.te_final > 0) and np.all(result.ti_final > 0)


def test_coupled_global_energy_balance():
    result = run_picard_multichannel(n_rho=51, s0_e=300.0, s0_i=20.0, tau_eq=1.0, tol=1e-7)
    assert result.metadata["converged"]
    r = result.rho
    h = r[1] - r[0]
    weights = r[:-1] * h
    weights[0] = h * h / 8
    src = 320 * np.exp(-((r - 0.3) ** 2) / 0.02)
    heating = weights @ src[:-1]
    outward = 0.0
    for t, chi in [
        (result.te_final, result.chi_e_profile),
        (result.ti_final, result.chi_i_profile),
    ]:
        outward += -0.5 * (r[-1] * chi[-1] + r[-2] * chi[-2]) * (t[-1] - t[-2]) / h
    assert outward == pytest.approx(heating, abs=1e-6)


def test_implicit_coupled_transient_preserves_stationary_solution():
    n = 31
    r = np.linspace(0, 1, n)
    src = np.exp(-((r - 0.3) ** 2) / 0.02)
    chis = [np.full(n, 0.2), np.full(n, 0.3)]
    t = stationary_solve(r, chis, r, [src, np.zeros(n)], [500, 400], exchange=150.0)
    out = solve_two_channel(
        n_rho=n,
        chi_e=chis[0],
        chi_i=chis[1],
        source_e=src,
        source_i=np.zeros(n),
        te_init=t[0],
        ti_init=t[1],
        n_steps=4,
        dt=0.1,
    )
    np.testing.assert_allclose(out["Te"], t[0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(out["Ti"], t[1], atol=1e-8, rtol=0)


@pytest.mark.parametrize("kw", [{"n_rho": 2}, {"tau_eq": 0}, {"density": -1}, {"sigma": 0}])
def test_invalid_physics_is_rejected(kw):
    with pytest.raises(ValueError):
        run_picard_multichannel(**kw)


@pytest.mark.parametrize("name", ["low_power", "mid_power", "high_power"])
def test_legacy_scenarios_regression(name):
    cfg = yaml.safe_load(
        (Path(__file__).parent / "fixtures/legacy_scenarios" / f"{name}.yaml").read_text()
    )
    result = run_one_scenario(cfg)
    assert result.metadata["converged"]
    assert result.metadata["final_residual"] <= cfg["picard"]["tol"]
    assert result.metadata["temperature_clips"] == 0


def test_crank_nicolson_second_order_in_time_against_matrix_exponential():
    from scipy.linalg import expm
    from scipy.special import j0, jn_zeros

    from tokamak_transport_lab.solver.steady import diffusion_operator

    n = 31
    rho = np.linspace(0, 1, n)
    initial = 500 + 10 * j0(jn_zeros(0, 1)[0] * rho)
    initial[-1] = 500.0
    operator = diffusion_operator(rho, np.ones(n), rho).toarray()
    exact = expm(0.2 * operator) @ initial
    errors = []
    for dt in (0.02, 0.01, 0.005):
        result = solve_two_channel(
            n_rho=n,
            te_init=initial,
            ti_init=initial,
            te_ped=500.0,
            ti_ped=500.0,
            s0_e=0.0,
            s0_i=0.0,
            density=0.0,
            dt=dt,
            n_steps=round(0.2 / dt),
        )
        errors.append(np.linalg.norm(result["Te"] - exact))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
