"""Stationary coupled temperatures with fully implicit exchange."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tokamak_transport_lab.integration.nonlinear import iterate
from tokamak_transport_lab.integration.picard import TransportModel


@dataclass
class MultichannelResult:
    rho: NDArray[np.float64]
    te_final: NDArray[np.float64]
    ti_final: NDArray[np.float64]
    chi_e_profile: NDArray[np.float64]
    chi_i_profile: NDArray[np.float64]
    residual_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def equilibration_source(te, ti, *, density=1.0, tau_eq=0.01):
    """Normalized exchange rate; see docs/physics.md for its convention."""
    if not np.isfinite(density) or density < 0 or not np.isfinite(tau_eq) or tau_eq <= 0:
        raise ValueError("density must be nonnegative and tau_eq strictly positive")
    return 1.5 * density * (np.asarray(te) - np.asarray(ti)) / tau_eq


def run_picard_multichannel(
    *,
    # Grid / physics
    n_rho: int = 100,
    te_ped: float = 500.0,
    ti_ped: float = 400.0,
    s0_e: float = 1.0,
    s0_i: float = 0.0,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
    # Coupling
    density: float = 1.0,
    tau_eq: float = 0.01,
    # Geometry
    v_prime_fn: Any = None,
    # Electron transport
    transport_model_e: TransportModel | None = None,
    transport_params_e: dict[str, Any] | None = None,
    # Ion transport
    transport_model_i: TransportModel | None = None,
    transport_params_i: dict[str, Any] | None = None,
    # Fallback
    fallback_model: TransportModel | None = None,
    fallback_params: dict[str, Any] | None = None,
    # Solver
    dt: float = 1e-4,
    theta: float = 0.5,
    sub_steps: int = 500,
    # Picard control
    max_iters: int = 60,
    tol: float = 1e-4,
    alpha0: float = 0.5,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    divergence_patience: int = 5,
    # Initial profiles
    te_init: NDArray[np.float64] | None = None,
    ti_init: NDArray[np.float64] | None = None,
) -> MultichannelResult:
    """Solve both stationary equations together, including exchange.

    Legacy dt/theta/sub_steps/divergence_patience arguments do not affect this
    stationary solve. No temperature clipping or implicit change of closure.
    """
    from tokamak_transport_lab.geometry.circular import v_prime
    from tokamak_transport_lab.physics.sources import gaussian_source
    from tokamak_transport_lab.transport.stiffness import chi_total

    exchange = float(equilibration_source(1.0, 0.0, density=density, tau_eq=tau_eq))
    rho = np.linspace(0, 1, n_rho)
    sources = [gaussian_source(rho, s0=s, rho_dep=rho_dep, sigma=sigma) for s in (s0_e, s0_i)]
    vp = (v_prime_fn or v_prime)(rho)
    initial = [
        te_init if te_init is not None else te_ped + 500 * (1 - rho * rho),
        ti_init if ti_init is not None else ti_ped + 300 * (1 - rho * rho),
    ]
    models = [transport_model_e or chi_total, transport_model_i or chi_total]
    params = [transport_params_e or {}, transport_params_i or {}]
    fallbacks = [fallback_params if fallback_params is not None else p for p in params]
    t, chi, res, alpha, meta = iterate(
        rho=rho,
        sources=sources,
        pedestals=[te_ped, ti_ped],
        vp=vp,
        initial=initial,
        models=models,
        params=params,
        fallback_model=fallback_model,
        fallback_params=fallbacks,
        exchange=exchange,
        max_iters=max_iters,
        tol=tol,
        alpha0=alpha0,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    meta.update(te_core_eV=float(t[0, 0]), ti_core_eV=float(t[1, 0]))
    return MultichannelResult(rho, t[0], t[1], chi[0], chi[1], res, alpha, meta)
