"""Coupled Crank-Nicolson evolution with exchange in the block operator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import eye
from scipy.sparse.linalg import factorized

from tokamak_transport_lab.solver.steady import balance_residual, coupled_operator


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
    """Time advance with fully implicit coupling and exact boundary rows.

    CN (theta=.5) is second order and A-stable, but can oscillate at large
    time steps; theta=1 is backward Euler. No temperature clipping is used.
    """
    if dt <= 0 or n_steps < 1 or not 0.5 <= theta <= 1 or sigma <= 0:
        raise ValueError("Invalid time controls or source width")
    if density < 0 or tau_eq <= 0 or not np.isfinite([density, tau_eq]).all():
        raise ValueError("Invalid exchange parameters")
    rho = np.linspace(0, 1, n_rho)
    chis = [np.broadcast_to(c, rho.shape).astype(float).copy() for c in (chi_e, chi_i)]
    vp = rho.copy() if v_prime is None else np.array(v_prime, dtype=float, copy=True)
    sources = [
        s0 * np.exp(-((rho - rho_dep) ** 2) / (2 * sigma * sigma))
        if src is None
        else np.asarray(src, dtype=float)
        for s0, src in ((s0_e, source_e), (s0_i, source_i))
    ]
    peds = [te_ped, ti_ped]
    t = np.array(
        [
            np.linspace(ped + rise, ped, n_rho) if init is None else np.array(init, copy=True)
            for ped, rise, init in zip(peds, [500, 300], [te_init, ti_init], strict=True)
        ]
    )
    if t.shape != (2, n_rho) or not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("Initial profiles must be finite, positive and match the grid")
    t[:, -1] = peds
    nu = 1.5 * density / tau_eq
    op = coupled_operator(rho, chis, vp, nu)
    implicit = eye(2 * n_rho, format="csc") - dt * theta * op
    explicit = eye(2 * n_rho, format="csr") + dt * (1 - theta) * op
    advance = factorized(implicit.tocsc())
    forcing = np.array(sources, copy=True)
    forcing[:, -1] = 0
    residuals = []
    for _ in range(n_steps):
        rhs = (explicit @ t.ravel()).reshape(t.shape) + dt * forcing
        rhs[:, -1] = peds
        new = advance(rhs.ravel()).reshape(t.shape)
        residuals.append(
            float(np.max(np.linalg.norm(new - t, axis=1) / np.linalg.norm(t, axis=1)))
        )
        t = new
    residual = float(
        np.max(np.linalg.norm(balance_residual(t, rho, chis, vp, sources, peds, nu), axis=1))
    )
    return {
        "rho": rho,
        "Te": t[0],
        "Ti": t[1],
        "chi_e": chis[0],
        "chi_i": chis[1],
        "source_e": sources[0],
        "source_i": sources[1],
        "residual_history": np.array(residuals),
        "steady_residual": residual,
        "converged": residual <= 1e-6,
        "time": dt * n_steps,
    }
