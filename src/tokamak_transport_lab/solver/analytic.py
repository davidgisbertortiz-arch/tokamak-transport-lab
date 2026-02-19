"""Semi-analytic steady-state reference via numerical quadrature.

For constant χ and V'(ρ) = ρ the steady-state satisfies::

    d/dρ [ρ χ dT/dρ] = -ρ S(ρ)

With Neumann at ρ = 0 and Dirichlet T(1) = T_ped, the solution is:

    I(ρ)  = ∫_0^ρ  ρ' S(ρ') dρ'
    dT/dρ = -(1 / (ρ χ)) I(ρ)          for ρ > 0
    T(ρ)  = T_ped + ∫_ρ^1 (1/(ρ'' χ)) I(ρ'') dρ''

This module computes the reference numerically using the trapezoidal rule
on a fine grid, with care near ρ = 0 where I(ρ)/ρ → 0 by symmetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def steady_state_reference(
    rho: NDArray[np.float64],
    chi0: float,
    s0: float,
    rho_dep: float,
    sigma: float,
    t_ped: float,
    *,
    n_quad: int = 4000,
) -> NDArray[np.float64]:
    """Compute the semi-analytic steady-state T(ρ) profile.

    Parameters
    ----------
    rho : array (N,)
        Evaluation grid in [0, 1].
    chi0 : float
        Constant diffusivity.
    s0, rho_dep, sigma : float
        Gaussian source parameters: S(ρ) = s0 exp(-(ρ-rho_dep)²/(2σ²)).
    t_ped : float
        Boundary temperature at ρ = 1.
    n_quad : int
        Number of quadrature points for integration (default 4000).

    Returns
    -------
    T_ref : array (N,)
        Steady-state temperature on *rho*.
    """
    # Fine quadrature grid
    rho_q = np.linspace(0.0, 1.0, n_quad)
    drho_q = rho_q[1] - rho_q[0]

    source_q = s0 * np.exp(-((rho_q - rho_dep) ** 2) / (2.0 * sigma**2))

    # I(ρ) = ∫_0^ρ  ρ' S(ρ') dρ'  (cumulative trapezoidal)
    integrand_I = rho_q * source_q
    I_of_rho = np.zeros(n_quad)
    for k in range(1, n_quad):
        I_of_rho[k] = I_of_rho[k - 1] + 0.5 * (integrand_I[k - 1] + integrand_I[k]) * drho_q

    # f(ρ) = I(ρ) / (ρ χ)  — the integrand for outer integral
    # Near ρ=0:  I(ρ) ~ ½ρ² S(0)  =>  I(ρ)/ρ → 0, so f(0) = S(0)/(2χ) via L'Hôpital.
    f = np.zeros(n_quad)
    f[0] = 0.5 * source_q[0] / chi0  # limiting value
    f[1:] = I_of_rho[1:] / (rho_q[1:] * chi0)

    # G(ρ) = ∫_ρ^1 f(ρ'') dρ''  — computed as G(ρ) = total - cumulative
    cumtrap_f = np.zeros(n_quad)
    for k in range(1, n_quad):
        cumtrap_f[k] = cumtrap_f[k - 1] + 0.5 * (f[k - 1] + f[k]) * drho_q
    total_f = cumtrap_f[-1]
    G_of_rho = total_f - cumtrap_f  # ∫_ρ^1

    T_fine = t_ped + G_of_rho

    # Interpolate onto requested grid
    T_ref = np.interp(rho, rho_q, T_fine)
    return T_ref
