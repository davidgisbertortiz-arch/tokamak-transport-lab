"""Headless GIF generation for README — no Streamlit required.

Generates animated frames showing Te/Ti profiles as a parameter sweeps,
then stitches them into a GIF using ``imageio`` (requires ``pip install -e '.[gif]'``).

Each frame is a **two-panel** figure:

* **Left** — Te(ρ) and Ti(ρ) profiles with a text overlay
  (swept param, Te0, final_residual, n_iters, converged).
* **Right** — Te0 vs swept parameter curve with a moving marker
  showing the current frame's position.

Usage::

    python -m scripts.make_readme_gif
    python -m scripts.make_readme_gif --config configs/scenarios/mid_power.yaml --n_frames 12
    python -m scripts.make_readme_gif --param S0_e --start 0.2 --end 12 --n_frames 12
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

# Softer stiffness → profiles visibly respond to S0_e sweep
_TRANSPORT_PARAMS: dict[str, float] = {
    "chi_s": 0.4,
    "a_over_LTe_crit": 2.0,
    "alpha_s": 1.5,
    "chi_neo": 0.01,
}


# ── single run helper ───────────────────────────────────────────


def _run_for_gif(
    *,
    s0_e: float = 1.0,
    te_ped: float = 250.0,
    ti_ped: float = 200.0,
    density: float = 1.0,
    tau_eq: float = 0.01,
    chi_s: float | None = None,
    a_over_LTe_crit: float | None = None,
    alpha_s: float | None = None,
    chi_neo: float | None = None,
    n_rho: int = 50,
    max_iters: int = 80,
    tol: float = 5e-4,
) -> MultichannelResult:
    """Lightweight run for GIF frames.

    Defaults are tuned so low-to-high P_heat sweeps produce visually
    distinct profiles with varying convergence behaviour.
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

_GIF_DPI = 130  # sharp enough for README embed


def render_frame(
    result: MultichannelResult,
    *,
    param_name: str = "S0_e",
    param_value: float = 1.0,
    te_ylim: tuple[float, float] | None = None,
    sweep_values: np.ndarray | None = None,
    sweep_te0s: np.ndarray | None = None,
    current_idx: int = 0,
) -> np.ndarray:
    """Render a two-panel frame and return as RGB uint8 array.

    Left panel: Te(ρ) + Ti(ρ) profiles with metadata overlay.
    Right panel: Te0 vs swept parameter with moving marker.
    """
    fig, (ax_prof, ax_trend) = plt.subplots(
        1,
        2,
        figsize=(10, 4.2),
        dpi=_GIF_DPI,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── left panel: profiles ─────────────────────────────────────
    ax_prof.plot(result.rho, result.te_final, "C0-", lw=2.2, label="Te")
    ax_prof.plot(result.rho, result.ti_final, "C3--", lw=2.2, label="Ti")
    ax_prof.set_xlabel("ρ")
    ax_prof.set_ylabel("Temperature [eV]")
    ax_prof.set_title("P_heat sweep — profiles")
    ax_prof.legend(loc="upper right", fontsize=8)
    ax_prof.grid(True, alpha=0.3)
    if te_ylim is not None:
        ax_prof.set_ylim(*te_ylim)

    # Text overlay
    n_iters = result.metadata.get("n_iters", len(result.residual_history))
    final_resid = result.residual_history[-1] if result.residual_history else float("nan")
    converged = result.metadata.get("converged", False)
    te0 = float(result.te_final[0])
    overlay = (
        f"{param_name} = {param_value:.2f}\n"
        f"Te0 = {te0:.0f} eV\n"
        f"residual = {final_resid:.2e}\n"
        f"n_iters = {n_iters}\n"
        f"converged = {converged}"
    )
    ax_prof.text(
        0.02,
        0.97,
        overlay,
        transform=ax_prof.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88},
    )

    # ── right panel: Te0 vs swept param ──────────────────────────
    if sweep_values is not None and sweep_te0s is not None:
        ax_trend.plot(sweep_values, sweep_te0s, "C0-", lw=1.8, label="Te(0)")
        ax_trend.plot(
            sweep_values[current_idx],
            sweep_te0s[current_idx],
            "o",
            color="C3",
            markersize=9,
            zorder=5,
        )
        ax_trend.set_xlabel(param_name)
        ax_trend.set_ylabel("Te(0) [eV]")
        ax_trend.set_title("Core temperature response")
        ax_trend.grid(True, alpha=0.3)
    else:
        ax_trend.set_visible(False)

    fig.suptitle("tokamak-transport-lab", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout()

    # Rasterise to numpy RGB array (drop alpha → smaller GIF)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf, dtype=np.uint8)
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


# ── GIF assembly ─────────────────────────────────────────────────


def generate_gif(
    *,
    param: str = "S0_e",
    start: float = 0.2,
    end: float = 12.0,
    n_frames: int = 12,
    out_path: str | pathlib.Path = "assets/demo.gif",
    duration_s: float = 0.45,
    base_kwargs: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Run a parameter sweep headlessly and stitch frames into a GIF.

    Parameters
    ----------
    param : str
        Parameter to sweep.  One of ``S0_e``, ``te_ped``, ``ti_ped``,
        ``density``, ``tau_eq``.
    start, end : float
        Sweep range.  Default 0.2 → 12.0 for ``S0_e`` produces
        clearly distinct profiles.
    n_frames : int
        Number of frames in the GIF (default 12).
    out_path : str or Path
        Destination file.
    duration_s : float
        Seconds per frame (default 0.45).
    base_kwargs : dict, optional
        Override default run parameters (from YAML scenario config).

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
        "te_ped": 250.0,
        "ti_ped": 200.0,
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

    # ── first pass: run all points, collect results ──────────────
    all_results: list[tuple[float, MultichannelResult]] = []
    te_max = 0.0
    for i, val in enumerate(values):
        kw = {**defaults, run_key: val}
        print(f"  frame {i + 1}/{n_frames}  {param}={val:.3f} …", end="", flush=True)
        res = _run_for_gif(**kw)
        meta = res.metadata
        print(
            f"  Te0={res.te_final[0]:.0f}  iters={meta.get('n_iters', '?')}"
            f"  conv={meta.get('converged', '?')}"
        )
        all_results.append((val, res))
        te_max = max(te_max, float(res.te_final.max()))

    te_ylim = (0.0, te_max * 1.15)

    # Pre-compute Te0 array for the trend panel
    sweep_vals = np.array([v for v, _ in all_results])
    sweep_te0s = np.array([float(r.te_final[0]) for _, r in all_results])

    # ── second pass: render frames ───────────────────────────────
    frames: list[np.ndarray] = []
    for idx, (val, res) in enumerate(all_results):
        frame = render_frame(
            res,
            param_name=param,
            param_value=val,
            te_ylim=te_ylim,
            sweep_values=sweep_vals,
            sweep_te0s=sweep_te0s,
            current_idx=idx,
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
