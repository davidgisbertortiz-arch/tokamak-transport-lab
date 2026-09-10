"""Headless rendering of verified synthetic stationary heating sweeps."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from tokamak_transport_lab.integration.multichannel_picard import (
    MultichannelResult,
)

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


# ── frame rendering ─────────────────────────────────────────────

_GIF_DPI = 150


def _style_ax(ax: plt.Axes) -> None:
    """Apply dark dashboard style to an axes."""
    ax.set_facecolor(_AX_FACECOLOR)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.title.set_fontsize(10.5)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, alpha=0.35, linewidth=0.5)


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
        fig = plt.figure(figsize=(13, 6.2), dpi=_GIF_DPI, facecolor=_FIG_FACECOLOR)

        # Layout: left panel spans full height, right column has 2 stacked
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[3, 2],
            height_ratios=[1, 1],
            hspace=0.42,
            wspace=0.30,
            left=0.06,
            right=0.97,
            top=0.88,
            bottom=0.14,
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
        ax_prof.set_xlabel("ρ", fontsize=9.5)
        ax_prof.set_ylabel("Temperature [eV]", fontsize=9.5)
        ax_prof.set_title("Temperature profiles")
        # Disable scientific / offset notation on y-axis
        _sfmt = ScalarFormatter(useOffset=False)
        _sfmt.set_scientific(False)
        ax_prof.yaxis.set_major_formatter(_sfmt)
        ax_prof.legend(
            loc="upper right",
            fontsize=8,
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
            tol_val = result.metadata["tol"]
            ax_conv.axhline(
                tol_val,
                color=_ACCENT,
                ls=":",
                lw=1,
                alpha=0.8,
            )
        ax_conv.set_xlabel("Nonlinear iteration", fontsize=8.5)
        ax_conv.set_ylabel("PDE residual", fontsize=8.5)
        ax_conv.set_title("Convergence")

        # ── Panel 3: χ profiles ──────────────────────────────────
        rho = result.rho
        ax_chi.plot(rho, result.chi_i_profile, color=_TI_COLOR, lw=1.5, label="χ_i total")
        ax_chi.plot(
            rho,
            result.chi_e_profile,
            color=_TEXT_COLOR,
            lw=1.0,
            ls=":",
            alpha=0.6,
            label="χ_e total",
        )
        ax_chi.set_xlabel("ρ", fontsize=8.5)
        ax_chi.set_ylabel("Normalized diffusivity χ", fontsize=8.5)
        ax_chi.set_title("Transport profiles")
        ax_chi.legend(
            loc="upper right",
            fontsize=7,
            facecolor=_AX_FACECOLOR,
            edgecolor=_GRID_COLOR,
            labelcolor=_TEXT_COLOR,
        )
        if chi_ylim is not None:
            ax_chi.set_ylim(*chi_ylim)

        # ── Suptitle + frame counter ────────────────────────────
        fig.suptitle(
            "tokamak-transport-lab  ·  normalized heating sweep",
            fontsize=13,
            fontweight="bold",
            color=_TEXT_COLOR,
            y=0.97,
        )
        # Frame counter badge (top-right)
        frame_label = f"frame {current_idx + 1}/{total_frames}"
        fig.text(
            0.97,
            0.97,
            frame_label,
            ha="right",
            va="top",
            fontsize=8.5,
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
            0.06,
            0.03,
            footer,
            ha="left",
            va="center",
            fontsize=8,
            fontfamily="monospace",
            color=_TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": _AX_FACECOLOR,
                "edgecolor": _GRID_COLOR,
                "alpha": 0.9,
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
    config=None,
    param="S0_e",
    start=300.0,
    end=1000.0,
    n_frames=8,
    out_path="assets/demo.gif",
    duration_s=0.6,
):
    """Solve a complete configuration at every sweep point and render it.

    Every frame must converge. Save the inputs and metrics alongside the GIF.
    """
    import json
    from copy import deepcopy

    import imageio.v3 as iio

    from tokamak_transport_lab.integration.config import run_config
    from tokamak_transport_lab.surrogate.profile_uq import family_config

    if n_frames < 2 or duration_s <= 0:
        raise ValueError("Need at least two frames and positive frame duration")
    config = deepcopy(config if config is not None else family_config())
    locations = {
        "S0_e": ("source", "S0_e"),
        "te_ped": ("bc", "Te_ped"),
        "ti_ped": ("bc", "Ti_ped"),
        "density": ("coupling", "density"),
        "tau_eq": ("coupling", "tau_eq"),
    }
    if param not in locations:
        raise ValueError(f"Unsupported sweep parameter: {param}")
    section, key = locations[param]
    all_results = []
    runs = []
    for value in np.linspace(start, end, n_frames):
        cfg = deepcopy(config)
        cfg.setdefault(section, {})[key] = float(value)
        result = run_config(cfg)
        if not result.metadata["converged"]:
            raise RuntimeError(f"GIF point {value} did not converge: {result.metadata}")
        all_results.append((float(value), result))
        runs.append({"value": float(value), "config": cfg, "metrics": result.metadata})
        print(
            f"{param}={value:.1f}: Te0={result.te_final[0]:.1f}, residual={result.metadata['final_residual']:.2e}",
            flush=True,
        )
    te_max = max(max(r.te_final.max(), r.ti_final.max()) for _, r in all_results)
    chi_max = max(max(r.chi_e_profile.max(), r.chi_i_profile.max()) for _, r in all_results)
    frames = [
        render_frame(
            r,
            param_name=param,
            param_value=value,
            te_ylim=(0.0, 1.1 * te_max),
            chi_ylim=(0.0, max(1.1 * chi_max, 0.02)),
            all_results=all_results,
            current_idx=i,
            total_frames=n_frames,
        )
        for i, (value, r) in enumerate(all_results)
    ]
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, extension=".gif", duration=int(duration_s * 1000), loop=0)
    out.with_suffix(".json").write_text(
        json.dumps({"description": "Normalized synthetic heating sweep", "runs": runs}, indent=2)
        + "\n"
    )
    return out
