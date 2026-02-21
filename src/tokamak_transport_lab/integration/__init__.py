"""Picard iteration integration loop sub-package."""

from tokamak_transport_lab.integration.convergence import (
    is_converged,
    relative_l2_residual,
)
from tokamak_transport_lab.integration.picard import PicardResult, run_picard
from tokamak_transport_lab.integration.relaxation import mix_profiles, update_alpha
from tokamak_transport_lab.integration.safeguards import (
    clamp_inputs,
    has_nonfinite,
    should_fallback,
)

__all__ = [
    "PicardResult",
    "clamp_inputs",
    "has_nonfinite",
    "is_converged",
    "mix_profiles",
    "relative_l2_residual",
    "run_picard",
    "should_fallback",
    "update_alpha",
]
