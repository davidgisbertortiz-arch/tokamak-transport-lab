"""Lightweight Latin Hypercube Sampling (no external dependencies).

Implements a simple centred-LHS scheme: divide each dimension into *n*
equal stns, place one sample at the centre of each stratum, then
randomly permute the assignment across dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def latin_hypercube(
    n_samples: int,
    n_dims: int,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate a Latin Hypercube sample on the unit hypercube [0, 1]^d.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_dims : int
        Number of dimensions.
    rng : numpy.random.Generator, optional
        Random generator.  If *None*, creates one with an arbitrary seed.

    Returns
    -------
    samples : ndarray (n_samples, n_dims)
        Each row is a point in [0, 1]^d.
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = np.empty((n_samples, n_dims), dtype=np.float64)

    for d in range(n_dims):
        # Central points within each stratum
        perm = rng.permutation(n_samples)
        samples[:, d] = (perm + 0.5) / n_samples

    return samples


def scale_samples(
    unit_samples: NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Scale unit-hypercube samples to specified bounds.

    Parameters
    ----------
    unit_samples : ndarray (n, d)
        Points in [0, 1]^d from :func:`latin_hypercube`.
    bounds : list of (lo, hi)
        Length *d* list of (lower, upper) for each dimension.

    Returns
    -------
    scaled : ndarray (n, d)
    """
    bounds_arr = np.asarray(bounds, dtype=np.float64)  # (d, 2)
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    return unit_samples * (hi - lo) + lo
