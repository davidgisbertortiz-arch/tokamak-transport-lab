"""Crank–Nicolson finite-difference 1-D diffusion solver.

Solves:
    ∂T/∂t = (1/V') ∂/∂ρ [V' χ ∂T/∂ρ] + S(ρ)

on a uniform grid ρ ∈ [0, 1] with:
    - Neumann (symmetry) BC at ρ = 0
    - Dirichlet (pedestal) BC at ρ = 1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _build_tridiag(
    rho: NDArray[np.float64],
    chi: NDArray[np.float64],
    v_prime: NDArray[np.float64],
    dt: float,
    theta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build tri-diagonal coefficients for the CN scheme.

    Returns (lower, diag, upper) arrays of length N (grid points).
    """
    n = len(rho)
    drho = rho[1] - rho[0]
    drho2 = drho * drho

    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)

    for i in range(1, n - 1):
        # Flux-conservative form: (1/V') d/drho [V' chi dT/drho]
        # Use half-grid values for V' * chi
        vp_chi_ip = 0.5 * (v_prime[i + 1] * chi[i + 1] + v_prime[i] * chi[i])
        vp_chi_im = 0.5 * (v_prime[i] * chi[i] + v_prime[i - 1] * chi[i - 1])

        coeff = dt * theta / (v_prime[i] * drho2)
        lower[i] = -coeff * vp_chi_im
        upper[i] = -coeff * vp_chi_ip
        diag[i] = 1.0 + coeff * (vp_chi_im + vp_chi_ip)

    # Dirichlet at rho = 1 (last point)
    diag[-1] = 1.0

    # Neumann (symmetry) at rho = 0 (first point):  dT/drho = 0
    # For symmetry at ρ=0 with V'(ρ)=ρ the equation simplifies.
    # Use L'Hôpital: (1/ρ) d/dρ [ρ χ dT/dρ] → 2 χ d²T/dρ² at ρ=0
    # Ghost-point approach: T[-1] = T[1]  =>  upper coeff doubles
    coeff0_lhop = dt * theta * 2.0 * chi[0] / drho2
    diag[0] = 1.0 + 2.0 * coeff0_lhop  # ghost: T[-1]=T[1] doubles the coefficient
    upper[0] = -2.0 * coeff0_lhop

    return lower, diag, upper


def _build_rhs(
    T: NDArray[np.float64],
    rho: NDArray[np.float64],
    chi: NDArray[np.float64],
    v_prime: NDArray[np.float64],
    source: NDArray[np.float64],
    dt: float,
    theta: float,
    t_ped: float,
) -> NDArray[np.float64]:
    """Build right-hand-side vector for the CN scheme."""
    n = len(rho)
    drho = rho[1] - rho[0]
    drho2 = drho * drho
    rhs = np.copy(T)

    one_m_theta = 1.0 - theta

    for i in range(1, n - 1):
        vp_chi_ip = 0.5 * (v_prime[i + 1] * chi[i + 1] + v_prime[i] * chi[i])
        vp_chi_im = 0.5 * (v_prime[i] * chi[i] + v_prime[i - 1] * chi[i - 1])
        coeff = dt * one_m_theta / (v_prime[i] * drho2)
        diff = coeff * (vp_chi_ip * (T[i + 1] - T[i]) - vp_chi_im * (T[i] - T[i - 1]))
        rhs[i] = T[i] + diff + dt * source[i]

    # Neumann at ρ=0 — L'Hôpital limit
    coeff0 = dt * one_m_theta * 2.0 * chi[0] / drho2
    rhs[0] = T[0] + 2.0 * coeff0 * (T[1] - T[0]) + dt * source[0]

    # Dirichlet at ρ=1
    rhs[-1] = t_ped

    return rhs


def _solve_tridiag(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tri-diagonal system via Thomas algorithm."""
    n = len(rhs)
    c = np.zeros(n)
    d = np.zeros(n)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def solve(
    *,
    n_rho: int = 100,
    chi0: float = 1.0,
    s0: float = 1.0,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
    t_ped: float = 500.0,
    dt: float = 1e-4,
    n_steps: int = 5000,
    theta: float = 0.5,
) -> dict[str, NDArray[np.float64]]:
    """Run the Crank–Nicolson solver to (approximate) steady state.

    Parameters
    ----------
    n_rho : int
        Number of radial grid points.
    chi0 : float
        Constant diffusivity [m²/s].
    s0 : float
        Source amplitude.
    rho_dep : float
        Source deposition centre in normalised radius.
    sigma : float
        Source Gaussian width.
    t_ped : float
        Pedestal (Dirichlet) temperature at ρ = 1 [eV].
    dt : float
        Time-step size.
    n_steps : int
        Number of time steps.
    theta : float
        Implicitness parameter (0.5 = Crank–Nicolson).

    Returns
    -------
    dict
        Keys: ``rho``, ``Te``, ``chi``, ``source``, ``residual_history``.
    """
    rho = np.linspace(0.0, 1.0, n_rho)
    chi = np.full(n_rho, chi0)
    source = s0 * np.exp(-((rho - rho_dep) ** 2) / (2.0 * sigma**2))

    # V'(ρ) = ρ  for circular geometry  (regularise at origin)
    v_prime = np.copy(rho)
    v_prime[0] = 1e-30  # avoid division by zero; handled via L'Hôpital in stencil

    # Initial condition: linear from some core value to pedestal
    T = np.linspace(t_ped + 500.0, t_ped, n_rho)

    lower, diag, upper = _build_tridiag(rho, chi, v_prime, dt, theta)

    residuals: list[float] = []
    for _step in range(n_steps):
        rhs = _build_rhs(T, rho, chi, v_prime, source, dt, theta, t_ped)
        T_new = _solve_tridiag(lower, diag, upper, rhs)
        res = float(np.linalg.norm(T_new - T) / max(np.linalg.norm(T), 1e-30))
        residuals.append(res)
        T = T_new

    return {
        "rho": rho,
        "Te": T,
        "chi": chi,
        "source": source,
        "residual_history": np.array(residuals),
    }
