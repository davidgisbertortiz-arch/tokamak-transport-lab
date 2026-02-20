"""Reproducibility: seed all RNGs in one call."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds for ``random``, ``numpy``, and (if available) ``torch``.

    Parameters
    ----------
    seed : int
        Master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass
