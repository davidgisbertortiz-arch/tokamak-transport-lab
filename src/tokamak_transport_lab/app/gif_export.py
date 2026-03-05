"""Headless GIF generation for README — no Streamlit required.

Generates portfolio-grade animated frames showing a P_heat sweep
dashboard, then stitches them into a GIF via ``imageio``.

Each frame is a **3-panel + footer** dashboard:

* **Top-left (big)** — Te(ρ) and Ti(ρ) profiles.  Previous sweep
  profiles shown as faint "ghost" lines; current frame is bold.
* **Top-right upper** — Convergence: residual vs iteration (log scale).
* **Top-right lower** — χ profiles: χ_turb + χ_neo decomposition.
* **Footer strip** — compact metadata (P_heat, Te₀, residual, iters, converged).

Usage::

    python -m scripts.make_readme_gif
    python -m scripts.make_readme_gif --n_frames 15
    python -m scripts.make_readme_gif --param P_heat --start 1 --end 80 --n_frames 15
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from tokamak_transport_lab.integration.multichannel_picard import (
    MultichannelResult,
    run_picard_multichannel,
)
from tokamak_transport_lab.integration.picard import _compute_a_over_lte
from tokamak_transport_lab.transport.stiffness import chi_total, chi_turbulent

# ── defaults ─────────────────────────────────────────────────────

# Demo transport: chi_neo dominates so Te0 responds linearly to P_heat.
# chi_s adds a small visible turbulent bump in the χ panel.
# tau_diff = L²/(4·chi_neo) ≈ 2.5  →  dt·sub_steps=0.02 per Picard
# iteration, ~50 iters = 1.0 ≈ tau_diff/2 → converges when starting
# from a flat (pedestal) initial profile.
_TRANSPORT_PARAMS: dict[str, float] = {
    "chi_s": 0.05,
    "a_over_LTe_crit": 2.0,
    "alpha_s": 1.5,
    "chi_neo": 0.1,
}

# ── dark style setup ────────────────────────────────────────────

_STYLE_CTX = "dark_background"
_FIG_FACECOLOR = "#181818"
_AX_FACECOLOR = "#222222"
_GRID_COLOR = "#444444"
_TEXT_COLOR = "#e0e0e0"
_TE_COLOR = "#4fc3f7"  # vivid cyan
_TI_COLOR = "#ff8a65"  # warm coral
_CHI_TURB_COLOR = "#81c784"  # green
_CHI_NEO_COLOR = "#ffb74d"  # amber
_ACCENT = "#e53935"  # red accent marker


# ── power-normalized source ─────────────────────────────────────


def _power_to_s0(
    p_heat: float,
    rho: np.ndarray,
    rho_dep: float = 0.3,
    sigma: float = 0.1,
) -> float:
    """Convert total heating power *P_heat* to Gaussian source amplitude *S0*.

    Solves  P_heat = ∫₀¹ S₀·exp(…) · V'(ρ) dρ  for S₀, where V'=ρ.
    """
    gauss = np.exp(-((rho - rho_dep) ** 2) / (2.0 * sigma**2))
    vprime = np.copy(rho)
    vprime[0] = max(vprime[0], 1e-30)
    norm = float(np.trapz(gauss * vprime, rho))
    return p_heat / max(norm, 1e-10)


# ── single run helper ───────────────────────────────────────────


def _run_for_gif(
    *,
    p_heat: float | None = None,
    s0_e: float = 1.0,
    te_ped: float = 250.0,
    ti_ped: float = 200.0,
    density: float = 0.2,
    tau_eq: float = 0.5,
    chi_s: float | None = None,
    a_over_LTe_crit: float | None = None,
    alpha_s: float | None = None,
    chi_neo: float | None = None,
    n_rho: int = 50,
    max_iters: int = 60,
    tol: float = 1e-3,
) -> MultichannelResult:
    """Lightweight run for GIF frames.

    If *p_heat* is given it is converted to *s0_e* via power
    normalisation (preferred).  Otherwise *s0_e* is used directly.

    Key design choices for demo stability:
    * Flat initial profiles at pedestal temperature — the solver
      *builds up* from T_ped instead of trying to diffuse a steep
      initial guess (which causes transient blow-up).
    * Weak equilibration (``tau_eq=0.5``) and low density (0.2) so
      the coupling source S_ei stays small.
    * CN step sized for convergence: ``dt=1e-3, sub_steps=300``
      (dt·sub_steps=0.3 per Picard iteration, ~12× τ_diff in 60 iters).
    """
    rho = np.linspace(0.0, 1.0, n_rho)

    # Power-normalised source amplitude
    if p_heat is not None:
        s0_e = _power_to_s0(p_heat, rho)

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

    # Start from flat profiles at pedestal T → solver builds up,
    # no overshoot from steep initial gradients.
    te_init = np.full(n_rho, te_ped)
    ti_init = np.full(n_rho, ti_ped)

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
        dt=1e-3,
        sub_steps=300,
        max_iters=max_iters,
        tol=tol,
        alpha0=0.5,
        te_init=te_init,
        ti_init=ti_init,
    )


# ── frame rendering ─────────────────────────────────────────────

_GIF_DPI = 130


def _style_ax(ax: plt.Axes) -> None:
    """Apply dark dashboard style to an axes."""
    ax.set_facecolor(_AX_FACECOLOR)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.title.set_fontsize(9)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, alpha=0.4, linewidth=0.5)


def render_frame(
    result: MultichannelResult,
    *,
    param_name: str = "S0_e",
    param_value: float = 1.0,
    te_ylim: tuple[float, float] | None = None,
    chi_ylim: tuple[float, float] | None = None,
    all_results: list[tuple[float, MultichannelResult]] | None = None,
    current_idx: int = 0,
    total_frames: int = 1,
) -> np.ndarray:
    """Render a 3-panel dashboard frame and return as RGB uint8 array."""
    with plt.style.context(_STYLE_CTX):
        fig = plt.figure(figsize=(11, 5.2), dpi=_GIF_DPI, facecolor=_FIG_FACECOLOR)

        # Layout: left panel spans full height, right column has 2 stacked
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[3, 2],
            height_ratios=[1, 1],
            hspace=0.38,
            wspace=0.32,
            left=0.07,
            right=0.96,
            top=0.90,
            bottom=0.15,
        )
        ax_prof = fig.add_subplot(gs[:, 0])  # full height left
        ax_conv = fig.add_subplot(gs[0, 1])  # top right
        ax_chi = fig.add_subplot(gs[1, 1])  # bottom right

        for ax in (ax_prof, ax_conv, ax_chi):
            _style_ax(ax)

        # ── Panel 1: profiles ────────────────────────────────────
        # Ghost lines for all sweep points (faint)
        if all_results is not None:
            for j, (_, r) in enumerate(all_results):
                if j == current_idx:
                    continue
                ax_prof.plot(r.rho, r.te_final, color=_TE_COLOR, alpha=0.12, lw=0.8)
                ax_prof.plot(r.rho, r.ti_final, color=_TI_COLOR, alpha=0.12, lw=0.8)

        # Current frame bold
        ax_prof.plot(result.rho, result.te_final, color=_TE_COLOR, lw=2.5, label="Te", zorder=5)
        ax_prof.plot(
            result.rho,
            result.ti_final,
            color=_TI_COLOR,
            lw=2.5,
            ls="--",
            label="Ti",
            zorder=5,
        )
        ax_prof.set_xlabel("ρ")
        ax_prof.set_ylabel("Temperature [eV]")
        ax_prof.set_title("Temperature profiles")
        # Disable scientific / offset notation on y-axis
        _sfmt = ScalarFormatter(useOffset=False)
        _sfmt.set_scientific(False)
        ax_prof.yaxis.set_major_formatter(_sfmt)
        ax_prof.legend(
            loc="upper right",
            fontsize=7,
            facecolor=_AX_FACECOLOR,
            edgecolor=_GRID_COLOR,
            labelcolor=_TEXT_COLOR,
        )
        if te_ylim is not None:
            ax_prof.set_ylim(*te_ylim)

        # ── Panel 2: convergence ─────────────────────────────────
        if result.residual_history:
            iters = np.arange(1, len(result.residual_history) + 1)
            ax_conv.semilogy(
                iters,
                result.residual_history,
                color=_TE_COLOR,
                lw=1.5,
            )
            tol_val = result.metadata.get("tol", 5e-4)
            if isinstance(tol_val, str):
                tol_val = 5e-4
            ax_conv.axhline(
                tol_val,
                color=_ACCENT,
                ls=":",
                lw=1,
                alpha=0.8,
            )
        ax_conv.set_xlabel("Picard iteration")
        ax_conv.set_ylabel("Residual")
        ax_conv.set_title("Convergence")

        # ── Panel 3: χ profiles ──────────────────────────────────
        rho = result.rho
        a_over_lte = _compute_a_over_lte(result.te_final, rho)
        p = _TRANSPORT_PARAMS
        chi_turb = chi_turbulent(
            a_over_lte,
            chi_s=p["chi_s"],
            a_over_LTe_crit=p["a_over_LTe_crit"],
            alpha_s=p["alpha_s"],
        )
        chi_neo_arr = np.full_like(rho, p["chi_neo"])

        ax_chi.plot(rho, chi_turb, color=_CHI_TURB_COLOR, lw=1.8, label="χ_turb")
        ax_chi.plot(rho, chi_neo_arr, color=_CHI_NEO_COLOR, lw=1.3, ls="--", label="χ_neo")
        ax_chi.plot(
            rho,
            result.chi_e_profile,
            color=_TEXT_COLOR,
            lw=1.0,
            ls=":",
            alpha=0.6,
            label="χ_e total",
        )
        ax_chi.set_xlabel("ρ")
        ax_chi.set_ylabel("Diffusivity χ")
        ax_chi.set_title("Transport profiles")
        ax_chi.legend(
            loc="upper right",
            fontsize=6.5,
            facecolor=_AX_FACECOLOR,
            edgecolor=_GRID_COLOR,
            labelcolor=_TEXT_COLOR,
        )
        if chi_ylim is not None:
            ax_chi.set_ylim(*chi_ylim)

        # ── Suptitle + frame counter ────────────────────────────
        fig.suptitle(
            "tokamak-transport-lab  ·  P_heat sweep",
            fontsize=11,
            fontweight="bold",
            color=_TEXT_COLOR,
            y=0.97,
        )
        # Frame counter badge (top-right)
        frame_label = f"frame {current_idx + 1}/{total_frames}"
        fig.text(
            0.96,
            0.97,
            frame_label,
            ha="right",
            va="top",
            fontsize=7.5,
            fontfamily="monospace",
            color=_ACCENT,
            fontweight="bold",
        )

        # NOT CONVERGED warning badge
        converged = result.metadata.get("converged", True)
        if not converged:
            fig.text(
                0.50,
                0.92,
                "NOT CONVERGED",
                ha="center",
                va="top",
                fontsize=9,
                fontfamily="monospace",
                color="#ffeb3b",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": _ACCENT,
                    "edgecolor": "#ffeb3b",
                    "alpha": 0.9,
                },
            )

        # ── Footer strip ─────────────────────────────────────────
        n_iters = result.metadata.get("n_iters", len(result.residual_history))
        final_resid = result.residual_history[-1] if result.residual_history else float("nan")
        converged = result.metadata.get("converged", False)
        te0 = float(result.te_final[0])
        ti0 = float(result.ti_final[0])
        footer = (
            f"{param_name}={param_value:.2f}    "
            f"Te\u2080={te0:.0f} eV    Ti\u2080={ti0:.0f} eV    "
            f"residual={final_resid:.2e}    "
            f"iters={n_iters}    "
            f"converged={'✓' if converged else '✗'}"
        )
        fig.text(
            0.07,
            0.03,
            footer,
            ha="left",
            va="center",
            fontsize=7.5,
            fontfamily="monospace",
            color=_TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": _AX_FACECOLOR,
                "edgecolor": _GRID_COLOR,
                "alpha": 0.85,
            },
        )

        # Rasterise
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        rgba = np.asarray(buf, dtype=np.uint8)
        rgb = rgba[:, :, :3].copy()
        plt.close(fig)
    return rgb


# ── GIF assembly ─────────────────────────────────────────────────


def generate_gif(
    *,
    param: str = "P_heat",
    start: float = 1.0,
    end: float = 15.0,
    n_frames: int = 12,
    out_path: str | pathlib.Path = "assets/demo.gif",
    duration_s: float = 0.45,
    base_kwargs: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Run a parameter sweep headlessly and stitch frames into a GIF.

    Parameters
    ----------
    param : str
        Parameter to sweep.  One of ``P_heat``, ``S0_e``, ``te_ped``,
        ``ti_ped``, ``density``, ``tau_eq``.
    start, end : float
        Sweep range.  Default 1.0 → 15.0 for ``P_heat`` produces
        clearly distinct profiles.
    n_frames : int
        Number of frames (default 12).
    out_path : str or Path
        Destination file.
    duration_s : float
        Seconds per frame (default 0.45).
    base_kwargs : dict, optional
        Override default run parameters (from YAML config).

    Returns
    -------
    pathlib.Path
        Path to the generated GIF.
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        msg = "imageio is required for GIF generation.  Install with:  pip install -e '.[gif]'"
        raise ImportError(msg) from exc

    values = np.linspace(start, end, n_frames)

    # ── defaults (may be overridden by base_kwargs) ──────────────
    # density & tau_eq are kept small so Te–Ti coupling doesn't
    # overwhelm the profiles during the sweep.
    defaults: dict[str, Any] = {
        "p_heat": 10.0,
        "s0_e": 1.0,
        "te_ped": 250.0,
        "ti_ped": 200.0,
        "density": 0.2,
        "tau_eq": 0.5,
    }
    if base_kwargs is not None:
        defaults.update(base_kwargs)

    _param_key = {
        "P_heat": "p_heat",
        "S0_e": "s0_e",
        "te_ped": "te_ped",
        "ti_ped": "ti_ped",
        "density": "density",
        "tau_eq": "tau_eq",
    }
    run_key = _param_key.get(param, param)

    # ── first pass: run all points ───────────────────────────────
    all_results: list[tuple[float, MultichannelResult]] = []
    te_max = 0.0
    chi_max = 0.0
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
        chi_max = max(chi_max, float(res.chi_e_profile.max()))

    te_ylim = (0.0, te_max * 1.15)
    chi_ylim = (0.0, max(chi_max * 1.2, 0.05))

    # ── summary ──────────────────────────────────────────────────
    te0_vals = [r.te_final[0] for _, r in all_results]
    n_conv = sum(1 for _, r in all_results if r.metadata.get("converged"))
    print(
        f"\n  ── sweep summary ──\n"
        f"  converged: {n_conv}/{n_frames}    "
        f"Te0 range: {min(te0_vals):.0f} – {max(te0_vals):.0f} eV\n"
    )

    # ── second pass: render frames ───────────────────────────────
    frames: list[np.ndarray] = []
    for idx, (val, res) in enumerate(all_results):
        frame = render_frame(
            res,
            param_name=param,
            param_value=val,
            te_ylim=te_ylim,
            chi_ylim=chi_ylim,
            all_results=all_results,
            current_idx=idx,
            total_frames=n_frames,
        )
        frames.append(frame)

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        out,
        frames,
        extension=".gif",
        duration=int(duration_s * 1000),
        loop=0,
        quantize=True,
    )
    return out
