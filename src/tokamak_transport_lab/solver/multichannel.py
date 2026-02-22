"""Two-channel Crank–Nicolson solver for coupled Te–Ti diffusion.

.. rubric:: Operator-splitting strategy

Each time step is split into two stages:

1. **Diffusion** — independent CN half-step for each channel:

   .. math::
       \\frac{\\partial T_e}{\\partial t} =
           \\frac{1}{V'} \\frac{\\partial}{\\partial \\rho}
           \\bigl[V'\\,\\chi_e\\,\\frac{\\partial T_e}{\\partial \\rho}\\bigr]
           + S_e(\\rho)

   (analogously for :math:`T_i` with :math:`\\chi_i,\\,S_i`).

2. **Exchange coupling** — analytic relaxation:

   .. math::
       \\frac{dT_e}{dt} = -\\nu_{ei}(T_e - T_i), \\qquad
       \\frac{dT_i}{dt} = +\\nu_{ei}(T_e - T_i)

   where :math:`\\nu_{ei} = \\tfrac{3}{2}\\,n / \\tau_{eq}`.  The sum
   :math:`T_e + T_i` is conserved; the difference decays exponentially:

   .. math::
       D(\\Delta t) = D(0)\\,\\exp(-2\\nu_{ei}\\,\\Delta t)

   This is exact, unconditionally stable, and energy-conserving.

.. rubric:: Boundary conditions

Both channels use:

- **Neumann** (symmetry) at ρ = 0:  ∂T/∂ρ = 0
- **Dirichlet** (pedestal) at ρ = 1:  Te = Te_ped,  Ti = Ti_ped
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokamak_transport_lab.solver.crank_nicolson import (
    _build_rhs,
    _build_tridiag,
    _solve_tridiag,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _exchange_step(
    te: NDArray[np.float64],
    ti: NDArray[np.float64],
    dt: float,
    density: float,
    tau_eq: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply analytic electron-ion exchange coupling over one time step.

    The coupling frequency is ν_ei = (3/2) n / τ_eq.  The sum
    Te + Ti is conserved and the difference Te − Ti decays as
    exp(−2 ν_ei Δt).

    Parameters
    ----------
    te, ti : array (N,)
        Temperature profiles *after* the diffusion sub-step.
    dt : float
        Time-step size.
    density : float
        Plasma density (normalised units).
    tau_eq : float
        Electron–ion equilibration time.

    Returns
    -------
    te_new, ti_new : array (N,)
        Profiles after exchange relaxation.
    """
    nu_ei = 1.5 * density / tau_eq
    decay = np.exp(-2.0 * nu_ei * dt)

    s = te + ti  # conserved
    d = (te - ti) * decay  # decaying difference

    te_new = 0.5 * (s + d)
    ti_new = 0.5 * (s - d)
    return te_new, ti_new


def solve_two_channel(
    *,
    n_rho: int = 100,
    # Diffusivity profiles (spatially varying)
    chi_e: NDArray[np.float64] | float = 1.0,
    chi_i: NDArray[np.float64] | float = 1.0,
    # Sources
    s0_e: float = 1.0,
    s0_i: float = 0.0,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
    source_e: NDArray[np.float64] | None = None,
    source_i: NDArray[np.float64] | None = None,
    # Boundary conditions (Dirichlet at ρ=1)
    te_ped: float = 500.0,
    ti_ped: float = 400.0,
    # Exchange coupling
    density: float = 1.0,
    tau_eq: float = 0.01,
    # Geometry
    v_prime: NDArray[np.float64] | None = None,
    # Solver control
    dt: float = 1e-4,
    n_steps: int = 5000,
    theta: float = 0.5,
    # Initial conditions
    te_init: NDArray[np.float64] | None = None,
    ti_init: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Time-advance coupled Te–Ti to (approximate) steady state.

    Uses Strang-like operator splitting: CN diffusion for each channel
    independently, then analytic exchange relaxation each time step.

    Parameters
    ----------
    n_rho : int
        Number of radial grid points on [0, 1].
    chi_e, chi_i : float or array (N,)
        Thermal diffusivity for electrons / ions.  If scalar, expanded
        to a constant array.
    s0_e, s0_i : float
        Gaussian source amplitudes (ignored if *source_e*/*source_i*
        are provided directly).
    rho_dep, sigma : float
        Gaussian source centre and width.
    source_e, source_i : array (N,), optional
        Pre-computed source profiles.  Override the Gaussian default.
    te_ped, ti_ped : float
        Pedestal temperatures at ρ = 1 (Dirichlet BCs) [eV].
    density : float
        Plasma density in units consistent with τ_eq.
    tau_eq : float
        Electron–ion equilibration time.  Smaller means stronger
        coupling between Te and Ti.
    v_prime : array (N,), optional
        Flux-surface volume element V'(ρ).  Defaults to circular
        geometry V' = ρ.
    dt : float
        Time-step size.
    n_steps : int
        Number of time steps.
    theta : float
        CN implicitness (0.5 = Crank–Nicolson).
    te_init, ti_init : array (N,), optional
        Initial temperature profiles.

    Returns
    -------
    dict
        Keys: ``rho``, ``Te``, ``Ti``, ``chi_e``, ``chi_i``,
        ``source_e``, ``source_i``, ``residual_history``.
        The residual is max(||ΔTe||/||Te||, ||ΔTi||/||Ti||).
    """
    rho = np.linspace(0.0, 1.0, n_rho)

    # ── diffusivity arrays ───────────────────────────────────────
    chi_e_arr = (
        np.full(n_rho, chi_e, dtype=np.float64)
        if np.ndim(chi_e) == 0
        else np.asarray(chi_e, dtype=np.float64)
    )
    chi_i_arr = (
        np.full(n_rho, chi_i, dtype=np.float64)
        if np.ndim(chi_i) == 0
        else np.asarray(chi_i, dtype=np.float64)
    )

    # ── source arrays ────────────────────────────────────────────
    if source_e is None:
        source_e = s0_e * np.exp(-((rho - rho_dep) ** 2) / (2.0 * sigma**2))
    else:
        source_e = np.asarray(source_e, dtype=np.float64)
    if source_i is None:
        source_i = s0_i * np.exp(-((rho - rho_dep) ** 2) / (2.0 * sigma**2))
    else:
        source_i = np.asarray(source_i, dtype=np.float64)

    # ── geometry ─────────────────────────────────────────────────
    if v_prime is None:
        vp = np.copy(rho)
        vp[0] = 1e-30
    else:
        vp = np.asarray(v_prime, dtype=np.float64)
        vp[0] = max(vp[0], 1e-30)

    # ── initial conditions ───────────────────────────────────────
    te = (
        np.array(te_init, dtype=np.float64, copy=True)
        if te_init is not None
        else np.linspace(te_ped + 500.0, te_ped, n_rho)
    )
    ti = (
        np.array(ti_init, dtype=np.float64, copy=True)
        if ti_init is not None
        else np.linspace(ti_ped + 300.0, ti_ped, n_rho)
    )

    # ── pre-build tri-diagonal systems (constant χ ⇒ reusable) ──
    lower_e, diag_e, upper_e = _build_tridiag(rho, chi_e_arr, vp, dt, theta)
    lower_i, diag_i, upper_i = _build_tridiag(rho, chi_i_arr, vp, dt, theta)

    # ── time advance ─────────────────────────────────────────────
    residuals: list[float] = []
    for _step in range(n_steps):
        # 1. CN diffusion — electron channel
        rhs_e = _build_rhs(te, rho, chi_e_arr, vp, source_e, dt, theta, te_ped)
        te_new = _solve_tridiag(lower_e, diag_e, upper_e, rhs_e)

        # 2. CN diffusion — ion channel
        rhs_i = _build_rhs(ti, rho, chi_i_arr, vp, source_i, dt, theta, ti_ped)
        ti_new = _solve_tridiag(lower_i, diag_i, upper_i, rhs_i)

        # 3. Exchange coupling (analytic implicit relaxation)
        te_new, ti_new = _exchange_step(te_new, ti_new, dt, density, tau_eq)

        # Re-enforce Dirichlet BCs (exchange step may perturb boundary)
        te_new[-1] = te_ped
        ti_new[-1] = ti_ped

        # Floor: keep temperatures positive (min 10 eV)
        np.maximum(te_new, 10.0, out=te_new)
        np.maximum(ti_new, 10.0, out=ti_new)

        # 4. Residual = max of both channel relative changes
        norm_te = max(float(np.linalg.norm(te)), 1e-30)
        norm_ti = max(float(np.linalg.norm(ti)), 1e-30)
        res_te = float(np.linalg.norm(te_new - te)) / norm_te
        res_ti = float(np.linalg.norm(ti_new - ti)) / norm_ti
        residuals.append(max(res_te, res_ti))

        te = te_new
        ti = ti_new

    return {
        "rho": rho,
        "Te": te,
        "Ti": ti,
        "chi_e": chi_e_arr,
        "chi_i": chi_i_arr,
        "source_e": source_e,
        "source_i": source_i,
        "residual_history": np.array(residuals),
    }
