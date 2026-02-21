"""Transport-model adapters for the Picard loop.

Provides a uniform ``(a_over_lte, **kwargs) → chi`` interface so the
Picard iterator can call either:

1. **Analytic** – wraps :func:`stiffness.chi_total` (no-op, it already
   matches the expected signature).
2. **Surrogate** – wraps a :class:`TransportMLP` (single or ensemble
   mean), converting numpy ↔ torch internally.
3. **SurrogateWithUQ** – like (2) but also stores conformal prediction
   intervals after each call (not used for convergence, only for
   post-hoc analysis).

All adapters are lightweight, CPU-only, and carry no hidden state beyond
the last UQ bands (option 3).

Limitations
-----------
- The MLP predicts *normalised heat flux* Qe/Q_gB = chi · (a/L_Te);
  the adapter inverts this to chi = Q_norm / a_over_lte, clamped to a
  minimum of ``chi_floor`` to avoid division-by-zero in low-gradient
  regions.
- Ensemble uncertainty is the member standard-deviation of Q_norm,
  *not* a calibrated chi uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ── analytic adapter (thin wrapper, mostly for symmetry) ─────────


def analytic_adapter(
    a_over_lte: NDArray[np.float64],
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Forward to :func:`~transport.stiffness.chi_total`.

    Accepts exactly the same kwargs (chi_s, a_over_LTe_crit, …).
    This exists so user code can treat analytic and surrogate identically.
    """
    from tokamak_transport_lab.transport.stiffness import chi_total

    return chi_total(a_over_lte, **kwargs)


# ── surrogate adapter ───────────────────────────────────────────


@dataclass
class SurrogateAdapter:
    """Wrap a :class:`TransportMLP` as a Picard-compatible callable.

    The MLP predicts Q_norm = chi · a/L_Te.  The adapter converts back
    to chi by dividing by a/L_Te (with a floor to avoid blow-up).

    Parameters
    ----------
    model : TransportMLP (or any Module with compatible forward)
        Trained surrogate.  Must be on CPU and in eval mode.
    chi_floor : float
        Minimum returned chi (avoids division by near-zero gradient).
    extra_features : dict[str, float]
        Additional scalar features appended to every sample to match the
        MLP's expected input dimension.  For example
        ``{"q": 1.4, "s_hat": 0.8, ...}``.  They are appended in the
        dict's insertion order.

    Usage
    -----
    >>> adapter = SurrogateAdapter(model=mlp)
    >>> chi = adapter(a_over_lte_array)
    """

    model: Any  # TransportMLP — avoid hard import of torch at module level
    chi_floor: float = 1e-4
    extra_features: dict[str, float] = field(default_factory=dict)

    def __call__(
        self,
        a_over_lte: NDArray[np.float64],
        **_kwargs: Any,
    ) -> NDArray[np.float64]:
        """Evaluate surrogate and return chi(rho)."""
        import torch

        a = np.asarray(a_over_lte, dtype=np.float64).ravel()
        n = len(a)

        # Build feature matrix: [a/L_Te, extra1, extra2, …]
        extras = np.array(list(self.extra_features.values()), dtype=np.float64)
        if extras.size:
            x = np.column_stack([a, np.tile(extras, (n, 1))])
        else:
            x = a[:, np.newaxis]

        with torch.no_grad():
            t_in = torch.tensor(x, dtype=torch.float32)
            q_norm = self.model(t_in).squeeze(-1).numpy().astype(np.float64)

        # chi = Q_norm / (a/L_Te),  floored
        a_safe = np.maximum(np.abs(a), self.chi_floor)
        chi = q_norm / a_safe
        return np.maximum(chi, self.chi_floor)


# ── surrogate + conformal UQ adapter ────────────────────────────


@dataclass
class SurrogateWithUQAdapter:
    """Surrogate adapter that also stores conformal prediction bands.

    After each ``__call__``, the attributes :attr:`last_lower` and
    :attr:`last_upper` hold the [lo, hi] chi bands.  These are *not*
    fed back into the Picard loop — they are stored for post-hoc
    analysis and plotting only.

    Parameters
    ----------
    model : TransportMLP
        Trained surrogate (eval mode, CPU).
    q_hat : float
        Conformal quantile (adjusted non-conformity score), obtained
        from :class:`~surrogate.conformal.SplitConformal`.
    chi_floor : float
        Minimum returned chi.
    extra_features : dict[str, float]
        Same as :class:`SurrogateAdapter`.
    """

    model: Any
    q_hat: float = 0.0
    chi_floor: float = 1e-4
    extra_features: dict[str, float] = field(default_factory=dict)
    last_lower: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    last_upper: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __call__(
        self,
        a_over_lte: NDArray[np.float64],
        **_kwargs: Any,
    ) -> NDArray[np.float64]:
        """Evaluate surrogate, store UQ bands, return point-estimate chi."""
        import torch

        a = np.asarray(a_over_lte, dtype=np.float64).ravel()
        n = len(a)

        extras = np.array(list(self.extra_features.values()), dtype=np.float64)
        if extras.size:
            x = np.column_stack([a, np.tile(extras, (n, 1))])
        else:
            x = a[:, np.newaxis]

        with torch.no_grad():
            t_in = torch.tensor(x, dtype=torch.float32)
            q_norm = self.model(t_in).squeeze(-1).numpy().astype(np.float64)

        a_safe = np.maximum(np.abs(a), self.chi_floor)
        chi = q_norm / a_safe

        # Conformal bands on Q_norm → propagate to chi
        q_lo = np.maximum(q_norm - self.q_hat, 0.0)
        q_hi = q_norm + self.q_hat

        self.last_lower = np.maximum(q_lo / a_safe, self.chi_floor)
        self.last_upper = np.maximum(q_hi / a_safe, self.chi_floor)

        return np.maximum(chi, self.chi_floor)
