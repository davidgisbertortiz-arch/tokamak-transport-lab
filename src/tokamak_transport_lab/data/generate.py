"""Generate synthetic transport datasets via LHS + stiffness model.

Produces NPZ files with input features and output labels suitable for
training / testing a transport surrogate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokamak_transport_lab.data.lhs import latin_hypercube, scale_samples
from tokamak_transport_lab.transport.stiffness import normalised_flux

if TYPE_CHECKING:
    from pathlib import Path

# Default LHS sampling bounds for 6 input dimensions
DEFAULT_BOUNDS: list[tuple[float, float]] = [
    (0.5, 12.0),   # a/L_Te
    (1.0, 5.0),    # q (safety factor)
    (0.1, 3.0),    # s_hat (magnetic shear)
    (0.01, 2.0),   # nu_star (collisionality)
    (0.5, 5.0),    # chi_s (stiffness coefficient)
    (2.0, 6.0),    # a/L_Te,crit (critical gradient)
]

FEATURE_NAMES: list[str] = [
    "a_over_LTe",
    "q",
    "s_hat",
    "nu_star",
    "chi_s",
    "a_over_LTe_crit",
]


def generate_dataset(
    n_samples: int,
    *,
    bounds: list[tuple[float, float]] | None = None,
    alpha_s: float = 1.5,
    chi_neo: float = 0.01,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a synthetic transport dataset.

    Parameters
    ----------
    n_samples : int
        Number of data points.
    bounds : list of (lo, hi), optional
        Per-dimension bounds.  Defaults to :data:`DEFAULT_BOUNDS`.
    alpha_s : float
        Stiffness exponent (held fixed, not sampled).
    chi_neo : float
        Neoclassical floor diffusivity.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : dict
        ``"X"`` — input features ``(n_samples, 6)``
        ``"y"`` — normalised flux ``Qe/Q_gB`` ``(n_samples,)``
        ``"feature_names"`` — list of 6 strings
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n_samples, len(bounds), rng=rng)
    X = scale_samples(unit, bounds)

    # Evaluate stiffness model for each sample
    a_over_LTe = X[:, 0]
    chi_s = X[:, 4]
    a_over_LTe_crit = X[:, 5]

    y = normalised_flux(
        a_over_LTe,
        chi_s=chi_s,
        a_over_LTe_crit=a_over_LTe_crit,
        alpha_s=alpha_s,
        chi_neo=chi_neo,
    )

    return {
        "X": X,
        "y": y,
        "feature_names": np.array(FEATURE_NAMES),
    }


def save_dataset(data: dict[str, np.ndarray], path: Path | str) -> None:
    """Save a dataset dict to compressed NPZ."""
    from pathlib import Path as _P

    path = _P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
