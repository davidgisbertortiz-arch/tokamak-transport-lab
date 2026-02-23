"""Headless GIF generation for README — no Streamlit required.

Generates animated frames showing Te/Ti profiles as a parameter sweeps,
then stitches them into a GIF using ``imageio`` (requires ``pip install -e '.[gif]'``).

Each frame includes a text overlay with the swept parameter value,
final residual, and Picard iteration count.

Usage::

    python -m scripts.make_readme_gif
    python -m scripts.make_readme_gif --config configs/scenarios/mid_power.yaml --n_frames 10
    python -m scripts.make_readme_gif --param S0_e --start 0.5 --end 8 --n_frames 12
    python -m scripts.make_readme_gif --out assets/demo.gif
"""

from __future__ import annotations

import pathlib
from typing import Any

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
    chi_s: float | None = None,
    a_over_LTe_crit: float | None = None,
    alpha_s: float | None = None,
    chi_neo: float | None = None,
    n_rho: int = 50,
    max_iters: int = 40,
    tol: float = 1e-3,
) -> MultichannelResult:
    """Lightweight run for GIF frames.

    Supports overriding transport parameters and solver settings
    when driven from a YAML scenario config.
    """
    params_e = {
        "chi_s": chi_s if chi_s is not None else _TRANSPORT_PARAMS["chi_s"],
        "a_over_LTe_crit": (
            a_over_LTe_crit
            if a_over_LTe_crit is not None
            else _TRANSPORT_PARAMS["a_over_LTe_crit"]
        ),
        "alpha_s": alpha_s if alpha_s is not None else _TRANSPORT_PARAMS["alpha_s"],
        "chi_neo": chi_neo if chi_neo is not None else _TRANSPORT_PARAMS["chi_neo"],
    }
    params_i = {**params_e, "chi_s": params_e["chi_s"] * 0.5}
    return run_picard_multichannel(
        n_rho=n_rho,
        te_ped=te_ped,
        ti_ped=ti_ped,
        s0_e=s0_e,
        s0_i=0.0,
        density=density,
        tau_eq=tau_eq,
        transport_model_e=chi_total,
        transport_params_e=params_e,
        transport_model_i=chi_total,
        transport_params_i=params_i,
        dt=1e-4,
        sub_steps=200,
        max_iters=max_iters,
        tol=tol,
        alpha0=0.4,
    )


# ── frame rendering ─────────────────────────────────────────────

_GIF_DPI = 90  # balance quality / file size (<5 MB)


def render_frame(
    result: MultichannelResult,
    *,
    param_name: str = "S0_e",
    param_value: float = 1.0,
    te_ylim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a single matplotlib frame and return as RGB uint8 array.

    The frame shows Te(ρ) and Ti(ρ) profiles with a text overlay
    containing the swept parameter value, final residual, and
    iteration count.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=_GIF_DPI)

    ax.plot(result.rho, result.te_final, "C0-", linewidth=2.2, label="Te")
    ax.plot(result.rho, result.ti_final, "C3--", linewidth=2.2, label="Ti")
    ax.set_xlabel("ρ")
    ax.set_ylabel("Temperature [eV]")
    ax.set_title("tokamak-transport-lab — P_heat sweep")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if te_ylim is not None:
        ax.set_ylim(*te_ylim)

    # ── text overlay ─────────────────────────────────────────────
    n_iters = result.metadata.get("n_iters", len(result.residual_history))
    final_resid = result.residual_history[-1] if result.residual_history else float("nan")
    overlay = (
        f"{param_name} = {param_value:.2f}\nresidual = {final_resid:.2e}\nn_iters = {n_iters}"
    )
    ax.text(
        0.02,
        0.97,
        overlay,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()

    # Rasterise to numpy RGB array (drop alpha → smaller GIF)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf, dtype=np.uint8)
    rgb = rgba[:, :, :3].copy()  # drop alpha channel
    plt.close(fig)
    return rgb


# ── GIF assembly ─────────────────────────────────────────────────


def generate_gif(
    *,
    param: str = "S0_e",
    start: float = 0.5,
    end: float = 6.0,
    n_frames: int = 10,
    out_path: str | pathlib.Path = "assets/demo.gif",
    duration_s: float = 0.6,
    base_kwargs: dict[str, Any] | None = None,
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
    base_kwargs : dict, optional
        Override default run parameters (from YAML scenario config).
        Keys may include ``s0_e``, ``te_ped``, ``ti_ped``, ``density``,
        ``tau_eq``, ``chi_s``, ``a_over_LTe_crit``, ``alpha_s``,
        ``chi_neo``, ``n_rho``, ``max_iters``, ``tol``.

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

    # ── defaults (may be overridden by base_kwargs) ──────────────
    defaults: dict[str, Any] = {
        "s0_e": 1.0,
        "te_ped": 500.0,
        "ti_ped": 400.0,
        "density": 1.0,
        "tau_eq": 0.01,
    }
    if base_kwargs is not None:
        defaults.update(base_kwargs)

    # Map sweep param name → _run_for_gif kwarg
    _param_key = {
        "S0_e": "s0_e",
        "te_ped": "te_ped",
        "ti_ped": "ti_ped",
        "density": "density",
        "tau_eq": "tau_eq",
    }
    run_key = _param_key.get(param, param)

    # ── first pass: run all points, track y-limits ───────────────
    all_results: list[tuple[float, MultichannelResult]] = []
    te_max = 0.0
    for i, val in enumerate(values):
        kw = {**defaults, run_key: val}
        print(f"  frame {i + 1}/{n_frames}  {param}={val:.3f} …", end="", flush=True)
        res = _run_for_gif(**kw)
        print(f"  iters={res.metadata.get('n_iters', '?')}")
        all_results.append((val, res))
        te_max = max(te_max, float(res.te_final.max()))

    te_ylim = (0.0, te_max * 1.15)

    # ── second pass: render frames ───────────────────────────────
    frames: list[np.ndarray] = []
    for val, res in all_results:
        frame = render_frame(
            res,
            param_name=param,
            param_value=val,
            te_ylim=te_ylim,
        )
        frames.append(frame)

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # imageio v3 API — quantize=True keeps palette small
    iio.imwrite(
        out,
        frames,
        extension=".gif",
        duration=int(duration_s * 1000),
        loop=0,
        quantize=True,
    )
    return out
