"""Crank-Nicolson time evolution on the conservative radial stencil.

This is a transient solver: always inspect `converged` before treating the
last profile as stationary. Stationary nonlinear runs use integration.picard.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import eye
from scipy.sparse.linalg import factorized

from tokamak_transport_lab.solver.steady import balance_residual, diffusion_operator


def solve(
    *,
    n_rho: int = 100,
    chi0: float = 1.0,
    s0: float = 1.0,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
    t_ped: float = 500.0,
    dt: float = 1e-3,
    n_steps: int = 5000,
    theta: float = 0.5,
    steady_tol: float = 1e-6,
) -> dict[str, NDArray[np.float64]]:
    """Advance exactly n_steps and report the final source-scaled PDE defect.

    chi0 is normalized diffusivity, s0 is eV per normalized time, and dt
    is normalized time. See docs/physics.md. No device power is inferred.
    """
    if dt <= 0 or n_steps < 1 or not 0.5 <= theta <= 1 or t_ped <= 0 or sigma <= 0:
        raise ValueError("Invalid time-step, step count, theta, pedestal or source width")
    rho = np.linspace(0, 1, n_rho)
    chi = np.full(n_rho, chi0)
    source = s0 * np.exp(-((rho - rho_dep) ** 2) / (2 * sigma * sigma))
    op = diffusion_operator(rho, chi, rho)
    implicit = eye(n_rho, format="csc") - dt * theta * op
    explicit = eye(n_rho, format="csr") + dt * (1 - theta) * op
    advance = factorized(implicit.tocsc())
    t = np.linspace(t_ped + 500, t_ped, n_rho)
    residuals = []
    forcing = source.copy()
    forcing[-1] = 0
    for _ in range(n_steps):
        rhs = explicit @ t + dt * forcing
        rhs[-1] = t_ped
        new = advance(rhs)
        residuals.append(float(np.linalg.norm(new - t) / max(np.linalg.norm(t), 1e-30)))
        t = new
    res = float(np.linalg.norm(balance_residual([t], rho, [chi], rho, [source], [t_ped])))
    return {
        "rho": rho,
        "Te": t,
        "chi": chi,
        "source": source,
        "residual_history": np.array(residuals),
        "steady_residual": res,
        "converged": res <= steady_tol,
        "time": dt * n_steps,
    }
