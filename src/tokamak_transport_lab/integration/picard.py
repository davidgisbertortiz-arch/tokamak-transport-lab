"""Stationary nonlinear transport solve (Picard with Newton rescue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from tokamak_transport_lab.integration.nonlinear import gradient_length, iterate


class TransportModel(Protocol):
    def __call__(self, a_over_lte: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]: ...


@dataclass
class PicardResult:
    rho: NDArray[np.float64]
    te_final: NDArray[np.float64]
    chi_profile: NDArray[np.float64]
    residual_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_compute_a_over_lte = gradient_length


def run_picard(
    *,
    # Grid / physics
    n_rho: int = 100,
    t_ped: float = 500.0,
    s0: float = 1.0,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
    # Geometry
    v_prime_fn: Any = None,
    # Transport
    transport_model: TransportModel | None = None,
    transport_params: dict[str, Any] | None = None,
    fallback_model: TransportModel | None = None,
    fallback_params: dict[str, Any] | None = None,
    # Solver
    dt: float = 1e-4,
    theta: float = 0.5,
    sub_steps: int = 500,
    # Picard control
    max_iters: int = 50,
    tol: float = 1e-4,
    alpha0: float = 0.5,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    divergence_patience: int = 5,
    # Initial profile
    te_init: NDArray[np.float64] | None = None,
) -> PicardResult:
    """Solve stationary diffusion; tolerance applies to the PDE balance.

    dt, theta, sub_steps and divergence_patience are retained for compatibility
    but do not control the stationary solve. Transients use solver.solve.
    """
    from tokamak_transport_lab.geometry.circular import v_prime
    from tokamak_transport_lab.physics.sources import gaussian_source
    from tokamak_transport_lab.transport.stiffness import chi_total

    rho = np.linspace(0, 1, n_rho)
    source = gaussian_source(rho, s0=s0, rho_dep=rho_dep, sigma=sigma)
    vp = (v_prime_fn or v_prime)(rho)
    initial = te_init if te_init is not None else t_ped + 500 * (1 - rho * rho)
    model = transport_model or chi_total
    params = transport_params or {}
    fallback = fallback_model
    # Analytic default needs no fallback; explicitly supplied model can opt in.
    t, chi, res, alpha, meta = iterate(
        rho=rho,
        sources=[source],
        pedestals=[t_ped],
        vp=vp,
        initial=[initial],
        models=[model],
        params=[params],
        fallback_model=fallback,
        fallback_params=[fallback_params if fallback_params is not None else params],
        exchange=0,
        max_iters=max_iters,
        tol=tol,
        alpha0=alpha0,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    return PicardResult(rho, t[0], chi[0], res, alpha, meta)
