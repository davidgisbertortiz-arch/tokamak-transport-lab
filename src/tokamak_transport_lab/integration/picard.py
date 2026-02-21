"""Picard iteration loop coupling transport model ↔ CN solver.

Algorithm
---------
1. Start from an initial Te(ρ) profile.
2. Compute local normalised gradient a/L_Te from Te.
3. Evaluate transport model → χ(ρ).
4. Run one CN time-block to obtain Te_new.
5. Under-relax: Te ← (1−α)·Te_old + α·Te_new.
6. Monitor ||Te_new − Te_old||₂ / ||Te_old||₂.
7. Adapt α; check convergence or divergence safeguards.
8. Repeat until converged, max_iters reached, or fallback triggered.

Limitations
-----------
- The "solver step" time-advances for ``sub_steps`` CN steps per Picard
  iteration; the Picard loop watches the *outer* residual only.
- Transport model must be a callable ``(a_over_LTe, **params) → chi``
  matching the :func:`~tokamak_transport_lab.transport.stiffness.chi_total`
  signature, OR a dict ``{"type": "surrogate", ...}`` for ML-based chi.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from tokamak_transport_lab.integration.convergence import (
    is_converged,
    relative_l2_residual,
)
from tokamak_transport_lab.integration.relaxation import mix_profiles, update_alpha
from tokamak_transport_lab.integration.safeguards import (
    has_nonfinite,
    should_fallback,
)
from tokamak_transport_lab.solver.crank_nicolson import (
    _build_rhs,
    _build_tridiag,
    _solve_tridiag,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ── transport model protocol ─────────────────────────────────────


class TransportModel(Protocol):
    """Callable returning chi(ρ) from a/L_Te."""

    def __call__(
        self,
        a_over_lte: NDArray[np.float64],
        **kwargs: Any,
    ) -> NDArray[np.float64]: ...


# ── result container ─────────────────────────────────────────────


@dataclass
class PicardResult:
    """Container for Picard iteration output."""

    rho: NDArray[np.float64]
    te_final: NDArray[np.float64]
    chi_profile: NDArray[np.float64]
    residual_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────


def _compute_a_over_lte(
    te: NDArray[np.float64],
    rho: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute normalised inverse gradient length a/L_Te = −(a/Te)·dTe/dr.

    For normalised coordinates where a = 1, this simplifies to
    −(1/Te)·dTe/dρ.  We use centred differences in the interior and
    one-sided at the boundaries.
    """
    n = len(rho)
    drho = rho[1] - rho[0] if n > 1 else 1.0
    grad = np.zeros(n)

    # Centred differences for interior
    grad[1:-1] = (te[2:] - te[:-2]) / (2.0 * drho)
    # One-sided at boundaries
    grad[0] = (te[1] - te[0]) / drho
    grad[-1] = (te[-1] - te[-2]) / drho

    # a/L_Te = -grad_Te / Te  (avoid division by zero)
    te_safe = np.maximum(np.abs(te), 1e-10)
    a_over_lte = -grad / te_safe

    # Clamp to avoid extreme values at edge
    return np.clip(a_over_lte, 0.0, 50.0)


def _solver_step(
    te: NDArray[np.float64],
    rho: NDArray[np.float64],
    chi: NDArray[np.float64],
    v_prime: NDArray[np.float64],
    source: NDArray[np.float64],
    t_ped: float,
    dt: float,
    theta: float,
    sub_steps: int,
) -> NDArray[np.float64]:
    """Run *sub_steps* of CN time advance with the given χ(ρ) profile.

    Unlike :func:`solver.solve`, this accepts an *externally provided*
    spatially-varying chi array so the Picard loop can update it.
    """
    # Regularise V' at origin
    vp = np.copy(v_prime)
    vp[0] = max(vp[0], 1e-30)

    lower, diag, upper = _build_tridiag(rho, chi, vp, dt, theta)
    t = np.copy(te)
    for _ in range(sub_steps):
        rhs = _build_rhs(t, rho, chi, vp, source, dt, theta, t_ped)
        t = _solve_tridiag(lower, diag, upper, rhs)
    return t


# ── main entry point ─────────────────────────────────────────────


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
    """Run the Picard iteration loop.

    Parameters
    ----------
    n_rho : int
        Number of radial grid points.
    t_ped : float
        Pedestal temperature at ρ = 1 [eV].
    s0, rho_dep, sigma : float
        Gaussian source parameters.
    v_prime_fn : callable(rho) → array, optional
        Geometry function.  Defaults to circular V'(ρ) = ρ.
    transport_model : callable
        Primary transport model:  ``(a_over_lte, **params) → chi``.
    transport_params : dict, optional
        Extra keyword arguments forwarded to *transport_model*.
    fallback_model : callable, optional
        Analytic fallback used when the primary model diverges or
        returns non-finite values.
    fallback_params : dict, optional
        Keyword arguments for *fallback_model*.
    dt : float
        CN time-step.
    theta : float
        CN implicitness (0.5 = Crank–Nicolson).
    sub_steps : int
        Number of CN sub-steps per Picard iteration.
    max_iters : int
        Maximum outer (Picard) iterations.
    tol : float
        Convergence tolerance on the relative L2 residual.
    alpha0, alpha_min, alpha_max : float
        Under-relaxation schedule parameters.
    divergence_patience : int
        Consecutive residual increases before triggering fallback.
    te_init : array, optional
        Initial temperature profile.  Defaults to a linear ramp.

    Returns
    -------
    PicardResult
        Contains rho, te_final, chi_profile, residual_history,
        alpha_history, and a metadata dict.
    """
    from tokamak_transport_lab.geometry.circular import v_prime as circ_vprime
    from tokamak_transport_lab.physics.sources import gaussian_source
    from tokamak_transport_lab.transport.stiffness import chi_total as default_chi

    t_start = time.perf_counter()

    # ── defaults ─────────────────────────────────────────────────
    rho = np.linspace(0.0, 1.0, n_rho)
    source = gaussian_source(rho, s0=s0, rho_dep=rho_dep, sigma=sigma)

    if v_prime_fn is None:
        vp = circ_vprime(rho)
    else:
        vp = v_prime_fn(rho)

    if transport_model is None:
        transport_model = default_chi
    if transport_params is None:
        transport_params = {}
    if fallback_model is None:
        fallback_model = default_chi
    if fallback_params is None:
        fallback_params = dict(transport_params)

    if te_init is not None:
        te = np.array(te_init, dtype=np.float64, copy=True)
    else:
        te = np.linspace(t_ped + 500.0, t_ped, n_rho)

    # ── iteration state ──────────────────────────────────────────
    alpha = alpha0
    residual_prev: float | None = None
    divergence_counter = 0
    converged = False
    used_fallback = False
    residual_history: list[float] = []
    alpha_history: list[float] = []
    chi = np.full(n_rho, 0.01)  # initial guess; will be overwritten

    active_model = transport_model
    active_params = transport_params

    for iteration in range(1, max_iters + 1):
        # 1. Compute local inputs
        a_over_lte = _compute_a_over_lte(te, rho)

        # 2. Evaluate transport model → chi(ρ)
        try:
            chi_new = np.asarray(
                active_model(a_over_lte, **active_params),
                dtype=np.float64,
            )
        except Exception:
            logger.warning(
                "Picard iter %d: transport model raised — falling back.",
                iteration,
            )
            chi_new = np.asarray(
                fallback_model(a_over_lte, **fallback_params),
                dtype=np.float64,
            )
            used_fallback = True

        # 3. Per-iteration NaN/Inf guard
        if has_nonfinite(chi_new):
            logger.warning(
                "Picard iter %d: chi contains NaN/Inf — using fallback.",
                iteration,
            )
            chi_new = np.asarray(
                fallback_model(a_over_lte, **fallback_params),
                dtype=np.float64,
            )
            used_fallback = True

        # Ensure chi is positive (numerical floor)
        chi = np.maximum(chi_new, 1e-6)

        # 4. Solver step
        te_new = _solver_step(
            te,
            rho,
            chi,
            vp,
            source,
            t_ped,
            dt,
            theta,
            sub_steps,
        )

        # 5. Under-relax
        te_mixed = mix_profiles(te, te_new, alpha)

        # 6. Residual
        residual = relative_l2_residual(te_mixed, te)

        # 7. Adapt alpha
        alpha = update_alpha(
            alpha,
            residual,
            residual_prev,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        residual_history.append(residual)
        alpha_history.append(alpha)

        # 8. Divergence tracking
        if residual_prev is not None and residual >= residual_prev:
            divergence_counter += 1
        else:
            divergence_counter = 0

        # 9. Fallback check
        if not used_fallback and should_fallback(divergence_counter, divergence_patience):
            logger.warning(
                "Picard iter %d: divergence patience exhausted (%d iters) "
                "— switching to fallback model permanently.",
                iteration,
                divergence_patience,
            )
            active_model = fallback_model
            active_params = fallback_params
            used_fallback = True
            divergence_counter = 0

        # 10. Convergence check
        if is_converged(residual, tol):
            converged = True
            te = np.asarray(te_mixed, dtype=np.float64)
            logger.info(
                "Picard converged at iteration %d  (residual=%.2e, tol=%.2e).",
                iteration,
                residual,
                tol,
            )
            break

        residual_prev = residual
        te = np.asarray(te_mixed, dtype=np.float64)

    wall_time = time.perf_counter() - t_start

    return PicardResult(
        rho=rho,
        te_final=te,
        chi_profile=chi,
        residual_history=residual_history,
        alpha_history=alpha_history,
        metadata={
            "n_iters": len(residual_history),
            "converged": converged,
            "used_fallback": used_fallback,
            "wall_time_s": wall_time,
            "final_residual": residual_history[-1] if residual_history else float("nan"),
        },
    )
