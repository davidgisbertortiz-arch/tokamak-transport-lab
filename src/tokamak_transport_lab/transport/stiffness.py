"""Analytic critical-gradient (stiffness) transport model.

The turbulent diffusivity follows:

.. math::
    \\chi_{\\text{turb}} = \\chi_s \\cdot \\max\\!\\left(0,\\;
        \\frac{a}{L_{T_e}} - \\frac{a}{L_{T_e,\\text{crit}}}
    \\right)^{\\alpha_s}

A neoclassical floor :math:`\\chi_{\\text{neo}}` guarantees non-zero transport
even below the critical gradient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def chi_turbulent(
    a_over_LTe: NDArray[np.float64] | float,
    *,
    chi_s: float = 1.0,
    a_over_LTe_crit: float = 3.0,
    alpha_s: float = 1.5,
) -> NDArray[np.float64]:
    """Compute turbulent diffusivity from the critical-gradient model.

    Parameters
    ----------
    a_over_LTe : array-like
        Normalised inverse gradient length :math:`a / L_{T_e}`.
    chi_s : float
        Stiffness coefficient (amplitude above threshold).
    a_over_LTe_crit : float
        Critical gradient threshold.
    alpha_s : float
        Stiffness exponent.

    Returns
    -------
    chi_turb : ndarray
        Turbulent diffusivity (same shape as *a_over_LTe*).
    """
    a_over_LTe = np.asarray(a_over_LTe, dtype=np.float64)
    if (
        not np.isfinite(a_over_LTe).all()
        or not np.isfinite(chi_s).all()
        or not np.isfinite(a_over_LTe_crit).all()
        or not np.isfinite(alpha_s).all()
        or np.any(np.asarray(chi_s) < 0)
        or np.any(np.asarray(a_over_LTe_crit) < 0)
        or np.any(np.asarray(alpha_s) <= 0)
    ):
        raise ValueError(
            "Closure parameters must be finite; stiffness/threshold nonnegative and exponent positive"
        )
    margin = np.maximum(0.0, a_over_LTe - a_over_LTe_crit)
    return chi_s * margin**alpha_s


def chi_total(
    a_over_LTe: NDArray[np.float64] | float,
    *,
    chi_s: float = 1.0,
    a_over_LTe_crit: float = 3.0,
    alpha_s: float = 1.5,
    chi_neo: float = 0.01,
) -> NDArray[np.float64]:
    """Total diffusivity = turbulent + neoclassical floor.

    Parameters
    ----------
    a_over_LTe : array-like
        Normalised inverse gradient length.
    chi_s, a_over_LTe_crit, alpha_s :
        Stiffness model parameters (see :func:`chi_turbulent`).
    chi_neo : float
        Prescribed normalized diffusivity floor (not a neoclassical calculation).

    Returns
    -------
    chi : ndarray
        Total diffusivity (same shape as *a_over_LTe*).
    """
    if not np.isfinite(chi_neo).all() or np.any(np.asarray(chi_neo) <= 0):
        raise ValueError("The normalized diffusivity floor must be strictly positive")
    return (
        chi_turbulent(
            a_over_LTe,
            chi_s=chi_s,
            a_over_LTe_crit=a_over_LTe_crit,
            alpha_s=alpha_s,
        )
        + chi_neo
    )


def normalised_flux(
    a_over_LTe: NDArray[np.float64] | float,
    *,
    chi_s: float = 1.0,
    a_over_LTe_crit: float = 3.0,
    alpha_s: float = 1.5,
    chi_neo: float = 0.01,
) -> NDArray[np.float64]:
    """Algebraic flux proxy chi * (a/L_T), retained as an educational helper.

    This is not a gyroBohm normalization. The current surrogate learns chi
    directly; temperature factors and dimensional energy flux are absent.
    """
    a_over_LTe = np.asarray(a_over_LTe, dtype=np.float64)
    chi = chi_total(
        a_over_LTe,
        chi_s=chi_s,
        a_over_LTe_crit=a_over_LTe_crit,
        alpha_s=alpha_s,
        chi_neo=chi_neo,
    )
    return chi * a_over_LTe
