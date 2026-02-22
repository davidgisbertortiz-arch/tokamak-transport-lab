"""Adaptive under-relaxation for the Picard iteration.

The mixing parameter alpha controls how aggressively the new iterate is
accepted:

    Te_{k+1} = (1 - alpha) * Te_k  +  alpha * Te_new

The schedule adapts alpha based on the residual trend:
- Residual decreasing → increase alpha (up to alpha_max).
- Residual increasing → halve alpha (down to alpha_min).
"""

from __future__ import annotations


def update_alpha(
    alpha: float,
    residual_curr: float,
    residual_prev: float | None,
    *,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    step_up: float = 0.05,
) -> float:
    """Return the adapted under-relaxation parameter.

    Parameters
    ----------
    alpha : float
        Current mixing parameter.
    residual_curr : float
        Residual at the current iteration.
    residual_prev : float or None
        Residual at the previous iteration.  ``None`` on the first
        iteration (alpha is returned unchanged).
    alpha_min, alpha_max : float
        Hard clamps on alpha.
    step_up : float
        Additive increment when residual decreases.

    Returns
    -------
    alpha_new : float
        Updated mixing parameter clamped to [alpha_min, alpha_max].
    """
    if residual_prev is None:
        return max(alpha_min, min(alpha, alpha_max))

    alpha_new = alpha + step_up if residual_curr < residual_prev else alpha / 2.0

    return max(alpha_min, min(alpha_new, alpha_max))


def mix_profiles(
    te_old: object,
    te_new: object,
    alpha: float,
) -> object:
    """Under-relax: Te = (1 - alpha) * Te_old + alpha * Te_new.

    Works with any array type supporting arithmetic (numpy, torch).
    """
    return (1.0 - alpha) * te_old + alpha * te_new  # type: ignore[operator]
