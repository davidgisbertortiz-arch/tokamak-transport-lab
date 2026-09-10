"""Synthetic diffusivity data with five causally active closure parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_transport_lab.data.lhs import latin_hypercube, scale_samples
from tokamak_transport_lab.transport.stiffness import chi_total

FEATURE_NAMES = ["a_over_LTe", "chi_s", "a_over_LTe_crit", "alpha_s", "chi_neo"]
DEFAULT_BOUNDS = [(0.0, 12.0), (0.1, 5.0), (1.0, 6.0), (1.0, 2.0), (0.001, 0.1)]


def generate_dataset(n_samples, *, bounds=None, seed=42, sampling="lhs"):
    """Sample the analytic closure; y is normalized chi, never device flux.

    Use iid sampling for conformal calibration/test exchangeability. LHS is
    useful for training but is not an exchangeable iid calibration design.
    """
    bounds = np.asarray(DEFAULT_BOUNDS if bounds is None else bounds, dtype=float)
    if n_samples < 1 or bounds.shape != (5, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Need positive sample count and five ordered nondegenerate bounds")
    rng = np.random.default_rng(seed)
    if sampling == "lhs":
        unit = latin_hypercube(n_samples, 5, rng=rng)
    elif sampling == "iid":
        unit = rng.random((n_samples, 5))
    else:
        raise ValueError("sampling must be lhs or iid")
    x = scale_samples(unit, bounds)
    y = chi_total(x[:, 0], **{name: x[:, i] for i, name in enumerate(FEATURE_NAMES) if i})
    return {
        "X": x,
        "y": y,
        "feature_names": np.array(FEATURE_NAMES),
        "bounds": bounds,
        "target": np.array("chi"),
        "seed": np.array(seed),
        "sampling": np.array(sampling),
        "schema_version": np.array(2),
    }


def save_dataset(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
