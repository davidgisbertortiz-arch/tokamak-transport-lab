"""Headless GIF generation for README — no Streamlit required.

Generates animated frames showing Te/Ti profiles as a parameter sweeps,
then stitches them into a GIF using ``imageio`` (requires ``pip install -e '.[gif]'``).

Usage::

    python -m scripts.make_readme_gif                       # default sweep
    python -m scripts.make_readme_gif --param S0_e --start 0.5 --end 8 --frames 12
    python -m scripts.make_readme_gif --out assets/demo.gif
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from tokamak_transport_lab.integration.multichannel_picard import (
    MultichannelResult,
    run_picard_multichannel,
)
from tokamak_transport_lab.transport.stiffness import chi_total

# ── defaults ─────────────────────────────────────────────────────

_TRANSPORT_PARAMS: dict[str, float] = {
    "chi_s": 1.0,
    "a_over_LTe_crit": 3.0,
    "alpha_s": 1.5,
    "chi_neo": 0.01,
}


# ── single run helper ───────────────────────────────────────────


def _run_for_gif(
    *,
    s0_e: float = 1.0,
    te_ped: float = 500.0,
    ti_ped: float = 400.0,
    density: float = 1.0,
    tau_eq: float = 0.01,
) -> MultichannelResult:
    """Lightweight run for GIF frames (small grid, fast convergence)."""
    params_i = {**_TRANSPORT_PARAMS, "chi_s": _TRANSPORT_PARAMS["chi_s"] * 0.5}
    return run_picard_multichannel(
        n_rho=50,
        te_ped=te_ped,
        ti_ped=ti_ped,
        s0_e=s0_e,
        s0_i=0.0,
        density=density,
        tau_eq=tau_eq,
        transport_model_e=chi_total,
        transport_params_e=_TRANSPORT_PARAMS,
        transport_model_i=chi_total,
        transport_params_i=params_i,
        dt=1e-4,
        sub_steps=200,
        max_iters=40,
        tol=1e-3,
        alpha0=0.4,
    )


# ── frame rendering ─────────────────────────────────────────────


def render_frame(
    result: MultichannelResult,
    *,
    label: str = "",
    te_ylim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a single matplotlib frame and return as RGBA uint8 array."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    ax1.plot(result.rho, result.te_final, "C0-", linewidth=2.2, label="Te")
    ax1.plot(result.rho, result.ti_final, "C3--", linewidth=2.2, label="Ti")
    ax1.set_xlabel("ρ")
    ax1.set_ylabel("Temperature [eV]")
    ax1.set_title(f"Profiles  {label}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if te_ylim is not None:
        ax1.set_ylim(*te_ylim)

    ax2.semilogy(result.residual_history, "k-", linewidth=1.5)
    ax2.set_xlabel("Picard iteration")
    ax2.set_ylabel("Relative L₂ residual")
    ax2.set_title("Convergence")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    # rasterise to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf, dtype=np.uint8).copy()
    plt.close(fig)
    return frame


# ── GIF assembly ─────────────────────────────────────────────────


def generate_gif(
    *,
    param: str = "S0_e",
    start: float = 0.5,
    end: float = 6.0,
    n_frames: int = 10,
    out_path: str | pathlib.Path = "assets/demo.gif",
    duration_s: float = 0.6,
) -> pathlib.Path:
    """Run a parameter sweep headlessly and stitch frames into a GIF.

    Parameters
    ----------
    param : str
        Parameter to sweep.  One of ``S0_e``, ``te_ped``, ``ti_ped``,
        ``density``, ``tau_eq``.
    start, end : float
        Sweep range.
    n_frames : int
        Number of frames in the GIF.
    out_path : str or Path
        Destination file.
    duration_s : float
        Seconds per frame.

    Returns
    -------
    pathlib.Path
        The path to the generated GIF file.
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        msg = "imageio is required for GIF generation.  Install with:  pip install -e '.[gif]'"
        raise ImportError(msg) from exc

    values = np.linspace(start, end, n_frames)
    frames: list[np.ndarray] = []

    # pre-run to determine y-axis limits
    all_results: list[tuple[float, MultichannelResult]] = []
    te_max = 0.0
    for val in values:
        kwargs: dict[str, float] = {
            "s0_e": 1.0,
            "te_ped": 500.0,
            "ti_ped": 400.0,
            "density": 1.0,
            "tau_eq": 0.01,
        }
        kwargs[param] = val
        res = _run_for_gif(**kwargs)
        all_results.append((val, res))
        te_max = max(te_max, float(res.te_final.max()))

    te_ylim = (0.0, te_max * 1.15)

    for val, res in all_results:
        label = f"({param}={val:.2f})"
        frame = render_frame(res, label=label, te_ylim=te_ylim)
        frames.append(frame)

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # imageio v3 API
    iio.imwrite(
        out,
        frames,
        extension=".gif",
        duration=int(duration_s * 1000),
        loop=0,
    )
    return out
