"""Ion-channel transport models.

Provides two strategies for computing the ion thermal diffusivity χ_i,
selected via the ``ti_transport_mode`` configuration key:

``scaled_qe`` (default / MVP)
    Scale the electron diffusivity by a constant factor:
    ``chi_i(ρ) = qi_factor × chi_e(ρ)``.
    This is the simplest physically-motivated choice: ion and electron
    heat fluxes share the same turbulent drive but with a channel-
    specific amplitude ratio.

``stiffness``
    Independent critical-gradient model for ions with its own
    parameters (chi_s_i, a/L_Ti_crit, alpha_i).  This is the same
    functional form as the electron stiffness model but evaluated on
    the ion temperature gradient a/L_Ti.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tokamak_transport_lab.transport.stiffness import chi_total as _chi_total_stiffness

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── scaled-Qe mode ──────────────────────────────────────────────


class ScaledQeTransport:
    """Ion transport = scaled copy of the electron diffusivity.

    At each Picard iteration the electron transport model is evaluated
    *first*, then this wrapper multiplies by ``qi_factor``.

    Parameters
    ----------
    electron_model : callable
        The electron transport model ``(a_over_lte, **kw) → chi_e``.
    electron_params : dict
        Keyword arguments forwarded to *electron_model*.
    qi_factor : float
        Scaling factor ``chi_i / chi_e``.  Default 1.0 (equal channels).
        Typical range: 0.5–1.5.
    """

    def __init__(
        self,
        electron_model: Any,
        electron_params: dict[str, Any] | None = None,
        qi_factor: float = 1.0,
    ) -> None:
        self.electron_model = electron_model
        self.electron_params = electron_params or {}
        self.qi_factor = qi_factor
        # Cache: updated each call via the electron gradient
        self._last_chi_e: NDArray[np.float64] | None = None

    def __call__(
        self,
        a_over_lti: NDArray[np.float64],
        *,
        _a_over_lte: NDArray[np.float64] | None = None,
        **_kwargs: Any,
    ) -> NDArray[np.float64]:
        """Return chi_i = qi_factor × chi_e.

        Parameters
        ----------
        a_over_lti : array
            Ion normalised gradient (not used in ``scaled_qe`` mode,
            but accepted for interface compatibility).
        _a_over_lte : array, optional
            Electron gradient, injected by the Picard loop so that
            chi_e can be evaluated.  If not provided, falls back to
            using *a_over_lti* as a proxy (less accurate but safe).
        """
        # Use electron gradient if available, otherwise fall back to ion
        grad = _a_over_lte if _a_over_lte is not None else a_over_lti
        chi_e = np.asarray(
            self.electron_model(grad, **self.electron_params),
            dtype=np.float64,
        )
        self._last_chi_e = chi_e
        return self.qi_factor * chi_e


# ── independent stiffness mode ───────────────────────────────────


def chi_total_ion(
    a_over_LTi: NDArray[np.float64] | float,
    *,
    chi_s: float = 0.5,
    a_over_LTe_crit: float = 4.0,
    alpha_s: float = 1.5,
    chi_neo: float = 0.01,
) -> NDArray[np.float64]:
    """Independent ion stiffness model.

    Same functional form as the electron stiffness model but with
    ion-specific default parameters.  The ``a_over_LTe_crit`` name is
    kept for interface compatibility with ``chi_total`` — it represents
    the critical ion gradient in this context.

    Parameters
    ----------
    a_over_LTi : array-like
        Normalised inverse gradient length for ions.
    chi_s : float
        Ion stiffness coefficient (typically smaller than electron).
    a_over_LTe_crit : float
        Ion critical gradient (named for compatibility).
    alpha_s : float
        Stiffness exponent.
    chi_neo : float
        Neoclassical floor.

    Returns
    -------
    chi_i : ndarray
    """
    return _chi_total_stiffness(
        a_over_LTi,
        chi_s=chi_s,
        a_over_LTe_crit=a_over_LTe_crit,
        alpha_s=alpha_s,
        chi_neo=chi_neo,
    )


# ── factory ──────────────────────────────────────────────────────


def make_ion_transport(
    mode: str = "stiffness",
    *,
    # scaled_qe params
    electron_model: Any = None,
    electron_params: dict[str, Any] | None = None,
    qi_factor: float = 1.0,
    # stiffness params (returned as a partial-like callable)
    ion_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Factory: build an ion transport model + params from config.

    Parameters
    ----------
    mode : str
        ``"scaled_qe"`` or ``"stiffness"``.
    electron_model : callable
        Required for ``scaled_qe`` mode.
    electron_params : dict
        Forwarded to *electron_model* in ``scaled_qe`` mode.
    qi_factor : float
        Scaling for ``scaled_qe``.
    ion_params : dict
        Stiffness parameters for ``stiffness`` mode.

    Returns
    -------
    (model, params) : tuple
        A callable and its kwargs, ready to pass to
        ``run_picard_multichannel(transport_model_i=model,
        transport_params_i=params)``.
    """
    if mode == "scaled_qe":
        if electron_model is None:
            from tokamak_transport_lab.transport.stiffness import chi_total

            electron_model = chi_total
        model = ScaledQeTransport(
            electron_model=electron_model,
            electron_params=electron_params,
            qi_factor=qi_factor,
        )
        return model, {}

    if mode == "stiffness":
        params = ion_params or {}
        return chi_total_ion, params

    msg = f"Unknown ti_transport_mode: {mode!r}. Use 'scaled_qe' or 'stiffness'."
    raise ValueError(msg)
