"""Parity benchmark: analytic vs. surrogate Picard solutions.

Runs the Picard loop twice — once with the analytic stiffness model and
once with a surrogate adapter — then computes relative error metrics and
returns a JSON-serialisable report.

The comparison is performed on the *same* scenario (grid, source, BCs)
so that any discrepancy is purely due to the transport model mismatch.

Limitations
-----------
- Only the steady-state Te profile is compared; transient dynamics are
  not benchmarked.
- The surrogate must already be trained and wrapped in a
  :class:`~transport.adapters.SurrogateAdapter`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tokamak_transport_lab.integration.picard import run_picard

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tokamak_transport_lab.integration.picard import TransportModel

logger = logging.getLogger(__name__)


def parity_report(
    *,
    surrogate_model: TransportModel,
    surrogate_params: dict[str, Any] | None = None,
    analytic_params: dict[str, Any] | None = None,
    picard_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run analytic and surrogate Picard loops and compare Te profiles.

    Parameters
    ----------
    surrogate_model : callable
        A Picard-compatible transport model (e.g.
        :class:`~transport.adapters.SurrogateAdapter`).
    surrogate_params : dict, optional
        Extra kwargs forwarded to the surrogate inside Picard.
    analytic_params : dict, optional
        Kwargs for the analytic ``chi_total`` model.  Defaults to
        ``{"chi_neo": 0.01}``.
    picard_kwargs : dict, optional
        Common keyword arguments forwarded to :func:`run_picard` for
        both runs (n_rho, t_ped, dt, sub_steps, max_iters, tol, …).

    Returns
    -------
    report : dict
        JSON-serialisable dictionary with keys:

        - ``max_rel_error`` – max |Te_sur − Te_ana| / Te_ana
        - ``mean_rel_error`` – mean of the same
        - ``rmse`` – root-mean-square absolute error
        - ``analytic_converged`` – bool
        - ``surrogate_converged`` – bool
        - ``analytic_n_iters`` – int
        - ``surrogate_n_iters`` – int
        - ``analytic_wall_s`` – float
        - ``surrogate_wall_s`` – float
    """
    from tokamak_transport_lab.transport.stiffness import chi_total

    pkw: dict[str, Any] = dict(picard_kwargs) if picard_kwargs else {}
    a_params: dict[str, Any] = dict(analytic_params) if analytic_params else {"chi_neo": 0.01}
    s_params: dict[str, Any] = dict(surrogate_params) if surrogate_params else {}

    # ── analytic run ─────────────────────────────────────────────
    logger.info("Running Picard with analytic model …")
    res_ana = run_picard(
        transport_model=chi_total,
        transport_params=a_params,
        **pkw,
    )

    # ── surrogate run ────────────────────────────────────────────
    logger.info("Running Picard with surrogate model …")
    res_sur = run_picard(
        transport_model=surrogate_model,
        transport_params=s_params,
        fallback_model=chi_total,
        fallback_params=a_params,
        **pkw,
    )

    # ── comparison metrics ───────────────────────────────────────
    te_ana: NDArray[np.float64] = res_ana.te_final
    te_sur: NDArray[np.float64] = res_sur.te_final

    te_ref = np.maximum(np.abs(te_ana), 1e-10)  # avoid /0
    abs_diff = np.abs(te_sur - te_ana)
    rel_diff = abs_diff / te_ref

    report: dict[str, Any] = {
        "max_rel_error": float(np.max(rel_diff)),
        "mean_rel_error": float(np.mean(rel_diff)),
        "rmse": float(np.sqrt(np.mean(abs_diff**2))),
        "analytic_converged": res_ana.metadata["converged"],
        "surrogate_converged": res_sur.metadata["converged"],
        "analytic_n_iters": res_ana.metadata["n_iters"],
        "surrogate_n_iters": res_sur.metadata["n_iters"],
        "analytic_wall_s": res_ana.metadata["wall_time_s"],
        "surrogate_wall_s": res_sur.metadata["wall_time_s"],
    }

    logger.info(
        "Parity: max_rel_error=%.4e  mean_rel_error=%.4e  rmse=%.4e",
        report["max_rel_error"],
        report["mean_rel_error"],
        report["rmse"],
    )

    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write *report* to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Report saved to %s", path)
