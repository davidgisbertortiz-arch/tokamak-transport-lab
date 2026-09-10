"""Discrete stationary diffusion and implicit electron-ion exchange.

All coefficients use the normalized convention in docs/physics.md.  The
boundary row imposes T(1); the axis row is the cylindrical symmetry limit.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import spsolve


def diffusion_operator(rho, chi, v_prime):
    """Return the diffusion matrix with a zero row at the Dirichlet boundary."""
    rho, chi, vp = map(lambda x: np.asarray(x, dtype=float), (rho, chi, v_prime))
    n = len(rho)
    if n < 3 or rho.shape != chi.shape or rho.shape != vp.shape:
        raise ValueError("rho, chi and V' must have matching shapes with at least 3 points")
    h = rho[1] - rho[0]
    if h <= 0 or not np.allclose(np.diff(rho), h) or not np.allclose(rho[[0, -1]], [0, 1]):
        raise ValueError("Use a uniform radial grid on [0, 1]")
    if not np.isfinite(chi).all() or np.any(chi <= 0):
        raise ValueError("Diffusivity must be finite and strictly positive")
    if not np.isfinite(vp).all() or np.any(vp[1:] <= 0) or abs(vp[0]) > 1e-12:
        raise ValueError("V' must vanish at the axis and be positive elsewhere")
    faces = 0.5 * (vp[:-1] * chi[:-1] + vp[1:] * chi[1:])
    lower, upper = np.zeros(n), np.zeros(n)
    lower[1:-1] = faces[:-1] / (vp[1:-1] * h * h)
    upper[1:-1] = faces[1:] / (vp[1:-1] * h * h)
    # Axis control volume: integral_0^(h/2) V' d rho = V'(h) h / 8.
    upper[0] = 8 * faces[0] / (vp[1] * h * h)
    return diags((lower[1:], -lower - upper, upper[:-1]), (-1, 0, 1), format="csr")


def coupled_operator(rho, chis, vp, exchange=0.0):
    """Assemble L including exchange on interior/axis rows only."""
    blocks = [diffusion_operator(rho, chi, vp) for chi in chis]
    if len(blocks) == 1:
        return blocks[0]
    if len(blocks) != 2 or not np.isfinite(exchange) or exchange < 0:
        raise ValueError("Expected one or two channels and nonnegative finite exchange")
    reaction = diags(np.r_[np.full(len(rho) - 1, exchange), 0.0])
    return bmat([[blocks[0] - reaction, reaction], [reaction, blocks[1] - reaction]], format="csr")


def stationary_solve(rho, chis, vp, sources, pedestals, exchange=0.0):
    """Solve the frozen-transport elliptic system, with exchange implicit."""
    n = len(rho)
    matrix = (-coupled_operator(rho, chis, vp, exchange)).tolil()
    rhs = np.asarray(sources, dtype=float).reshape(-1).copy()
    for c, ped in enumerate(pedestals):
        row = (c + 1) * n - 1
        matrix[row, row] = 1.0
        rhs[row] = ped
    return spsolve(matrix.tocsc(), rhs).reshape(len(chis), n)


def balance_residual(temperatures, rho, chis, vp, sources, pedestals, exchange=0.0):
    """PDE defect divided by a fixed source scale, independent of relaxation.

    Each channel uses max(||S||_2, 1) as its scale (normalized eV/time).
    The boundary defect uses the same scale. No iterate-size denominator.
    """
    t = np.asarray(temperatures)
    defect = (coupled_operator(rho, chis, vp, exchange) @ t.ravel()).reshape(t.shape)
    defect += np.asarray(sources)
    defect[:, -1] = t[:, -1] - np.asarray(pedestals)
    scales = np.maximum(np.linalg.norm(np.asarray(sources)[:, :-1], axis=1), 1.0)
    return defect / scales[:, None]
