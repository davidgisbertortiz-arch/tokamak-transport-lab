"""Coupled Te–Ti Picard iteration with electron-ion equilibration.

Extends the single-channel Picard loop to solve a two-channel system:

    ∂Te/∂t = (1/V') ∂/∂ρ [V' χ_e ∂Te/∂ρ] + Se − S_ei
    ∂Ti/∂t = (1/V') ∂/∂ρ [V' χ_i ∂Ti/∂ρ] + Si + S_ei

where the electron-ion equilibration source is:

    S_ei = (3/2) n (Te − Ti) / τ_eq

At each Picard iteration the coupling term is evaluated using the
*previous* profiles (lagged coupling), which is natural for fixed-point
iteration and converges reliably when under-relaxed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from tokamak_transport_lab.integration.convergence import (
    is_converged,
    relative_l2_residual,
)
from tokamak_transport_lab.integration.picard import (
    TransportModel,
    _compute_a_over_lte,
    _solver_step,
)
from tokamak_transport_lab.integration.relaxation import mix_profiles, update_alpha
from tokamak_transport_lab.integration.safeguards import has_nonfinite

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ── result container ─────────────────────────────────────────────


@dataclass
class MultichannelResult:
    """Container for coupled Te–Ti Picard iteration output."""

    rho: NDArray[np.float64]
    te_final: NDArray[np.float64]
    ti_final: NDArray[np.float64]
    chi_e_profile: NDArray[np.float64]
    chi_i_profile: NDArray[np.float64]
    residual_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── equilibration source ─────────────────────────────────────────


def equilibration_source(
    te: NDArray[np.float64],
    ti: NDArray[np.float64],
    *,
    density: float = 1.0,
    tau_eq: float = 0.01,
) -> NDArray[np.float64]:
    """Compute S_ei = (3/2) n (Te − Ti) / τ_eq.

    This is positive when Te > Ti (energy flows from electrons to ions).

    Parameters
    ----------
    te, ti : array (N,)
        Electron / ion temperature profiles [eV].
    density : float
        Plasma density in units consistent with the source term.
        In these normalised units a value of 1.0 works with ``tau_eq``
        controlling the coupling strength.
    tau_eq : float
        Electron-ion equilibration time.  Smaller → stronger coupling.

    Returns
    -------
    S_ei : array (N,)
        Exchange power density.  Positive means electrons lose energy
        to ions.
    """
    return 1.5 * density * (te - ti) / tau_eq


# ── main entry point ─────────────────────────────────────────────


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
    """Run coupled Te–Ti Picard iteration.

    Parameters
    ----------
    n_rho : int
        Number of radial grid points.
    te_ped, ti_ped : float
        Pedestal temperatures at ρ = 1 for electrons / ions [eV].
    s0_e, s0_i : float
        Gaussian source amplitudes for electrons / ions.
    rho_dep, sigma : float
        Source deposition centre and width.
    density : float
        Plasma density (normalised units).
    tau_eq : float
        Electron-ion equilibration time.
    v_prime_fn : callable, optional
        Geometry function returning V'(ρ).
    transport_model_e, transport_model_i : callable, optional
        Transport models for electron / ion channels.
    transport_params_e, transport_params_i : dict, optional
        Parameters forwarded to each transport model.
    fallback_model, fallback_params : callable / dict, optional
        Analytic fallback when the primary model fails.
    dt, theta, sub_steps : float / int
        CN solver parameters.
    max_iters : int
        Maximum Picard iterations.
    tol : float
        Convergence tolerance on the max of both channel residuals.
    alpha0, alpha_min, alpha_max : float
        Under-relaxation schedule.
    divergence_patience : int
        Consecutive residual increases before fallback.
    te_init, ti_init : array, optional
        Initial temperature profiles.

    Returns
    -------
    MultichannelResult
        Contains rho, te_final, ti_final, chi profiles, histories,
        and metadata.
    """
    from tokamak_transport_lab.geometry.circular import v_prime as circ_vprime
    from tokamak_transport_lab.physics.sources import gaussian_source
    from tokamak_transport_lab.transport.stiffness import chi_total as default_chi

    t_start = time.perf_counter()

    # ── defaults ─────────────────────────────────────────────────
    rho = np.linspace(0.0, 1.0, n_rho)
    source_e = gaussian_source(rho, s0=s0_e, rho_dep=rho_dep, sigma=sigma)
    source_i = gaussian_source(rho, s0=s0_i, rho_dep=rho_dep, sigma=sigma)

    vp = circ_vprime(rho) if v_prime_fn is None else v_prime_fn(rho)

    if transport_model_e is None:
        transport_model_e = default_chi
    if transport_params_e is None:
        transport_params_e = {}
    if transport_model_i is None:
        transport_model_i = default_chi
    if transport_params_i is None:
        transport_params_i = {}
    if fallback_model is None:
        fallback_model = default_chi
    if fallback_params is None:
        fallback_params = dict(transport_params_e)

    if te_init is not None:
        te = np.array(te_init, dtype=np.float64, copy=True)
    else:
        te = np.linspace(te_ped + 500.0, te_ped, n_rho)

    if ti_init is not None:
        ti = np.array(ti_init, dtype=np.float64, copy=True)
    else:
        ti = np.linspace(ti_ped + 300.0, ti_ped, n_rho)

    # ── iteration state ──────────────────────────────────────────
    alpha = alpha0
    residual_prev: float | None = None
    divergence_counter = 0
    converged = False
    used_fallback = False
    residual_history: list[float] = []
    alpha_history: list[float] = []
    chi_e = np.full(n_rho, 0.01)
    chi_i = np.full(n_rho, 0.01)

    for iteration in range(1, max_iters + 1):
        # 1. Compute normalised gradients for each channel
        a_over_lte = _compute_a_over_lte(te, rho)
        a_over_lti = _compute_a_over_lte(ti, rho)  # same formula, different profile

        # 2. Evaluate transport models → chi_e(ρ) and chi_i(ρ)
        try:
            chi_e_new = np.asarray(
                transport_model_e(a_over_lte, **transport_params_e),
                dtype=np.float64,
            )
        except Exception:
            logger.warning("Iter %d: electron transport raised — fallback.", iteration)
            chi_e_new = np.asarray(fallback_model(a_over_lte, **fallback_params), dtype=np.float64)
            used_fallback = True

        try:
            chi_i_new = np.asarray(
                transport_model_i(a_over_lti, **transport_params_i),
                dtype=np.float64,
            )
        except Exception:
            logger.warning("Iter %d: ion transport raised — fallback.", iteration)
            chi_i_new = np.asarray(fallback_model(a_over_lti, **fallback_params), dtype=np.float64)
            used_fallback = True

        # 3. NaN/Inf guard
        if has_nonfinite(chi_e_new):
            logger.warning("Iter %d: chi_e NaN/Inf — fallback.", iteration)
            chi_e_new = np.asarray(fallback_model(a_over_lte, **fallback_params), dtype=np.float64)
            used_fallback = True
        if has_nonfinite(chi_i_new):
            logger.warning("Iter %d: chi_i NaN/Inf — fallback.", iteration)
            chi_i_new = np.asarray(fallback_model(a_over_lti, **fallback_params), dtype=np.float64)
            used_fallback = True

        chi_e = np.maximum(chi_e_new, 1e-6)
        chi_i = np.maximum(chi_i_new, 1e-6)

        # 4. Compute coupling term (lagged: uses current Te, Ti)
        s_ei = equilibration_source(te, ti, density=density, tau_eq=tau_eq)

        # 5. Solve electron channel: Se − S_ei (electrons lose energy)
        source_e_eff = source_e - s_ei
        te_new = _solver_step(te, rho, chi_e, vp, source_e_eff, te_ped, dt, theta, sub_steps)

        # 6. Solve ion channel: Si + S_ei (ions gain energy)
        source_i_eff = source_i + s_ei
        ti_new = _solver_step(ti, rho, chi_i, vp, source_i_eff, ti_ped, dt, theta, sub_steps)

        # Floor: temperatures must remain positive (min 10 eV)
        te_new = np.maximum(te_new, 10.0)
        ti_new = np.maximum(ti_new, 10.0)

        # 7. Under-relax both channels
        te_mixed = np.asarray(mix_profiles(te, te_new, alpha), dtype=np.float64)
        ti_mixed = np.asarray(mix_profiles(ti, ti_new, alpha), dtype=np.float64)

        # 8. Combined residual = max of both channel residuals
        res_te = relative_l2_residual(te_mixed, te)
        res_ti = relative_l2_residual(ti_mixed, ti)
        residual = max(res_te, res_ti)

        # 9. Adapt alpha
        alpha = update_alpha(
            alpha,
            residual,
            residual_prev,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        residual_history.append(residual)
        alpha_history.append(alpha)

        # 10. Divergence tracking
        if residual_prev is not None and residual >= residual_prev:
            divergence_counter += 1
        else:
            divergence_counter = 0

        if not used_fallback and divergence_counter >= divergence_patience:
            logger.warning(
                "Iter %d: divergence patience exhausted — permanent fallback.",
                iteration,
            )
            transport_model_e = fallback_model
            transport_params_e = dict(fallback_params)
            transport_model_i = fallback_model
            transport_params_i = dict(fallback_params)
            used_fallback = True
            divergence_counter = 0

        # 11. Convergence check
        if is_converged(residual, tol):
            converged = True
            te = te_mixed
            ti = ti_mixed
            logger.info(
                "Multichannel Picard converged at iter %d (res=%.2e).",
                iteration,
                residual,
            )
            break

        residual_prev = residual
        te = te_mixed
        ti = ti_mixed

    wall_time = time.perf_counter() - t_start

    return MultichannelResult(
        rho=rho,
        te_final=te,
        ti_final=ti,
        chi_e_profile=chi_e,
        chi_i_profile=chi_i,
        residual_history=residual_history,
        alpha_history=alpha_history,
        metadata={
            "n_iters": len(residual_history),
            "converged": converged,
            "used_fallback": used_fallback,
            "wall_time_s": wall_time,
            "final_residual": (residual_history[-1] if residual_history else float("nan")),
            "te_core_eV": float(te[0]),
            "ti_core_eV": float(ti[0]),
        },
    )
