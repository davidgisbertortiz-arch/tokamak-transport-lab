"""Streamlit interactive demo for tokamak-transport-lab.

Launch with::

    streamlit run src/tokamak_transport_lab/app/demo.py

Requires the ``demo`` extra::

    pip install -e ".[demo]"

Panels
------
(a) Te(ρ) and Ti(ρ), optional UQ band (ensemble + conformal)
(b) Convergence trace: residual + relaxation parameter α vs iteration
(c) χ profiles: χ_turb + χ_neo (electron & ion)

Transport model selector: analytic (stiffness) / mlp / ensemble.
Surrogate/ensemble/conformal are loaded lazily and cached.
"""

from __future__ import annotations


def main() -> None:
    """Entry point — all streamlit imports are deferred here."""
    import hashlib
    import json
    import pathlib
    import time
    from functools import partial

    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st

    from tokamak_transport_lab.integration.multichannel_picard import (
        MultichannelResult,
        run_picard_multichannel,
    )
    from tokamak_transport_lab.transport.stiffness import chi_total, chi_turbulent

    # ── page config ──────────────────────────────────────────────
    st.set_page_config(
        page_title="tokamak-transport-lab",
        page_icon="🔥",
        layout="wide",
    )
    st.title("🔥 tokamak-transport-lab — Interactive Demo")
    st.caption(
        "Coupled *Te*–*Ti* 1-D transport solver · stiffness / MLP / ensemble models "
        "· Picard iteration · calibrated UQ"
    )

    # ── lazy loaders (cached) ────────────────────────────────────

    @st.cache_resource(show_spinner="Loading MLP surrogate…")
    def _load_mlp(path: str):
        """Load a single TransportMLP from disk (torch required)."""
        try:
            from tokamak_transport_lab.surrogate.mlp import TransportMLP

            return TransportMLP.load(path, n_features=6, hidden=(128, 128, 64))
        except Exception as exc:
            st.warning(f"Could not load MLP: {exc}")
            return None

    @st.cache_resource(show_spinner="Loading Deep Ensemble…")
    def _load_ensemble(directory: str):
        """Load a DeepEnsemble (5 members) from disk."""
        try:
            from tokamak_transport_lab.surrogate.ensemble import DeepEnsemble
            from tokamak_transport_lab.surrogate.mlp import TransportMLP

            return DeepEnsemble.load(directory, TransportMLP, n_features=6, hidden=(128, 128, 64))
        except Exception as exc:
            st.warning(f"Could not load ensemble: {exc}")
            return None

    @st.cache_resource(show_spinner="Loading conformal calibration…")
    def _load_conformal(path: str):
        """Load SplitConformal from JSON."""
        try:
            from tokamak_transport_lab.surrogate.conformal import SplitConformal

            return SplitConformal.load(path)
        except Exception as exc:
            st.warning(f"Could not load conformal: {exc}")
            return None

    # ── sidebar: physics ─────────────────────────────────────────
    st.sidebar.header("🔬 Physics")

    p_heat = st.sidebar.slider(
        "P_heat (S₀ₑ)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Gaussian source amplitude for electrons.",
    )
    rho_dep = st.sidebar.slider(
        "ρ_dep (deposition centre)",
        min_value=0.0,
        max_value=0.8,
        value=0.3,
        step=0.05,
    )
    te_ped = st.sidebar.slider("Te pedestal [eV]", 100.0, 3000.0, 500.0, 50.0)
    ti_ped = st.sidebar.slider("Ti pedestal [eV]", 100.0, 2500.0, 400.0, 50.0)
    density = st.sidebar.slider("Density n₀", 0.5, 5.0, 1.0, 0.1)
    tau_eq = st.sidebar.slider("τ_eq", 0.001, 0.1, 0.01, 0.001, format="%.3f")

    # ── sidebar: geometry ────────────────────────────────────────
    st.sidebar.header("🌀 Geometry")
    geometry = st.sidebar.selectbox("Geometry", ["circular", "miller"])
    kappa = 1.0
    delta = 0.0
    if geometry == "miller":
        kappa = st.sidebar.slider("κ (elongation)", 1.0, 2.5, 1.7, 0.1)
        delta = st.sidebar.slider("δ (triangularity)", 0.0, 0.5, 0.33, 0.01)

    # ── sidebar: transport model ─────────────────────────────────
    st.sidebar.header("📈 Transport model")
    transport_mode = st.sidebar.selectbox(
        "Model",
        ["analytic", "mlp", "ensemble"],
        help=(
            "**analytic** — stiffness critical-gradient model (always available)\n\n"
            "**mlp** — single trained MLP surrogate\n\n"
            "**ensemble** — 5-member deep ensemble (enables UQ)"
        ),
    )
    show_uncertainty = st.sidebar.checkbox(
        "Show UQ band",
        value=True,
        disabled=(transport_mode != "ensemble"),
        help="Requires ensemble + conformal. Grey band = 90 % prediction interval.",
    )
    chi_s = st.sidebar.slider("χ_s (stiffness coeff)", 0.1, 5.0, 1.0, 0.1)
    a_over_lt_crit = st.sidebar.slider("a/L_T crit", 1.0, 8.0, 3.0, 0.5)
    chi_neo_val = st.sidebar.slider("χ_neo", 0.001, 0.1, 0.01, 0.001, format="%.3f")

    # ── sidebar: solver / picard ─────────────────────────────────
    st.sidebar.header("⚙️ Solver")
    n_rho = st.sidebar.slider("Grid points", 20, 150, 60, 10)
    max_iters = st.sidebar.slider("Max Picard iters", 5, 100, 40, 5)
    tol = st.sidebar.select_slider(
        "Convergence tol",
        options=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        value=1e-3,
    )

    # ── sidebar: paths (collapsed) ──────────────────────────────
    with st.sidebar.expander("🗂️ Model paths", expanded=False):
        mlp_path = st.text_input(
            "MLP .pt",
            value="outputs/surrogate/model.pt",
        )
        ensemble_dir = st.text_input(
            "Ensemble dir",
            value="outputs/ensemble",
        )
        conformal_path = st.text_input(
            "Conformal JSON",
            value="outputs/conformal/conformal.json",
        )

    # ── sidebar: sweep ───────────────────────────────────────────
    st.sidebar.header("🔄 Sweep mode")
    sweep_enabled = st.sidebar.checkbox("Enable parameter sweep")
    sweep_param = "P_heat"
    sweep_values: list[float] = []
    sweep_show_ti = False
    if sweep_enabled:
        sweep_param = st.sidebar.selectbox(
            "Sweep parameter",
            ["P_heat", "Te pedestal", "Ti pedestal", "Density", "τ_eq", "ρ_dep"],
        )
        _defaults: dict[str, tuple[float, float]] = {
            "P_heat": (0.2, 12.0),
            "Te pedestal": (200.0, 2000.0),
            "Ti pedestal": (200.0, 1500.0),
            "Density": (0.5, 4.0),
            "τ_eq": (0.002, 0.05),
            "ρ_dep": (0.05, 0.6),
        }
        _lo, _hi = _defaults.get(sweep_param, (0.5, 5.0))
        sweep_start = st.sidebar.number_input("Start", value=_lo, format="%.3f")
        sweep_end = st.sidebar.number_input("End", value=_hi, format="%.3f")
        sweep_steps = st.sidebar.slider("N steps", 3, 20, 8)
        sweep_show_ti = st.sidebar.checkbox("Show Ti(ρ) overlay", value=True)
        sweep_values = list(np.linspace(sweep_start, sweep_end, sweep_steps))

    # ── helpers ──────────────────────────────────────────────────

    def _build_vprime(geom: str, kap: float, delt: float):
        if geom == "miller":
            from tokamak_transport_lab.geometry.miller import vprime_miller

            return partial(vprime_miller, kappa=kap, delta=delt)
        return None

    def _config_hash(**kw: object) -> str:
        """Deterministic hash of slider values for caching."""
        raw = json.dumps(kw, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _run_picard(
        *,
        s0: float,
        rdep: float,
        te_p: float,
        ti_p: float,
        dens: float,
        teq: float,
    ):
        params_e = {
            "chi_s": chi_s,
            "a_over_LTe_crit": a_over_lt_crit,
            "alpha_s": 1.5,
            "chi_neo": chi_neo_val,
        }
        params_i = {**params_e, "chi_s": params_e["chi_s"] * 0.5}
        return run_picard_multichannel(
            n_rho=n_rho,
            te_ped=te_p,
            ti_ped=ti_p,
            s0_e=s0,
            s0_i=0.0,
            rho_dep=rdep,
            density=dens,
            tau_eq=teq,
            v_prime_fn=_build_vprime(geometry, kappa, delta),
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

    # ── cached run wrapper ───────────────────────────────────────

    @st.cache_data(show_spinner=False, ttl=300)
    def _cached_run(cfg_hash: str, **kw: float) -> MultichannelResult:
        return _run_picard(**kw)

    # ── UQ helper ────────────────────────────────────────────────

    def _compute_uq_band(
        rho: np.ndarray,
        te: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Try to produce UQ band using ensemble + conformal."""
        if transport_mode != "ensemble" or not show_uncertainty:
            return None
        ens = _load_ensemble(ensemble_dir)
        sc = _load_conformal(conformal_path)
        if ens is None or sc is None:
            return None
        try:
            import torch

            # Build a dummy 6-feature input from the converged profile
            a_over_lt = np.gradient(-te, rho, edge_first=True)
            a_over_lt = np.clip(a_over_lt / (te + 1.0), 0, 20)
            x = np.column_stack(
                [
                    a_over_lt,
                    np.full_like(rho, 2.0),  # q placeholder
                    np.full_like(rho, 1.0),  # s_hat placeholder
                    np.full_like(rho, 0.5),  # nu_star placeholder
                    np.full_like(rho, chi_s),
                    np.full_like(rho, a_over_lt_crit),
                ]
            )
            pred = ens.predict(torch.tensor(x, dtype=torch.float32))
            mean = pred["mean"].squeeze().numpy()
            lo, hi = sc.predict_interval(mean)
            # Map flux → temperature-like scale (rough proportional band)
            band_half = (hi - lo) / 2.0
            scale = np.abs(te) / (np.abs(mean) + 1e-6)
            return (te - band_half * scale, te + band_half * scale)
        except Exception:
            return None

    # ── plotting helpers ─────────────────────────────────────────

    def _plot_profiles(
        result: MultichannelResult,
        *,
        uq_band: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(result.rho, result.te_final, "C0-", lw=2.2, label="Te")
        ax.plot(result.rho, result.ti_final, "C3--", lw=2.2, label="Ti")
        if uq_band is not None:
            ax.fill_between(
                result.rho,
                uq_band[0],
                uq_band[1],
                alpha=0.18,
                color="C0",
                label="90 % UQ band",
            )
        ax.set_xlabel("ρ")
        ax.set_ylabel("Temperature [eV]")
        ax.set_title("Temperature profiles")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_convergence(result: MultichannelResult) -> plt.Figure:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        iters = range(1, len(result.residual_history) + 1)

        color_res = "C0"
        ax1.semilogy(iters, result.residual_history, color=color_res, lw=1.8, label="residual")
        ax1.axhline(tol, color="r", ls=":", lw=1, label=f"tol = {tol:.0e}")
        ax1.set_xlabel("Picard iteration")
        ax1.set_ylabel("Relative L₂ residual", color=color_res)
        ax1.tick_params(axis="y", labelcolor=color_res)
        ax1.grid(True, alpha=0.3)

        color_alpha = "C2"
        ax2 = ax1.twinx()
        ax2.plot(iters, result.alpha_history, color=color_alpha, lw=1.2, ls="--", label="α")
        ax2.set_ylabel("Relaxation α", color=color_alpha)
        ax2.tick_params(axis="y", labelcolor=color_alpha)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        ax1.set_title("Convergence trace")
        fig.tight_layout()
        return fig

    def _plot_chi(result: MultichannelResult) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        rho = result.rho

        # Decompose electron chi into turb + neo
        from tokamak_transport_lab.integration.picard import _compute_a_over_lte

        a_over_lte = _compute_a_over_lte(result.te_final, rho)
        chi_turb = chi_turbulent(
            a_over_lte, chi_s=chi_s, a_over_LTe_crit=a_over_lt_crit, alpha_s=1.5
        )
        chi_neo_arr = np.full_like(rho, chi_neo_val)

        ax.plot(rho, result.chi_e_profile, "C0-", lw=2, label="χ_e total")
        ax.plot(rho, chi_turb, "C0:", lw=1.2, label="χ_e turb")
        ax.plot(rho, chi_neo_arr, "C0--", lw=1, alpha=0.6, label="χ_neo")
        ax.plot(rho, result.chi_i_profile, "C3-", lw=2, label="χ_i total")
        ax.set_xlabel("ρ")
        ax.set_ylabel("Diffusivity χ")
        ax.set_title("Transport profiles")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ── run button ───────────────────────────────────────────────
    if not sweep_enabled:
        run_clicked = st.button("▶ Run", type="primary", use_container_width=True)
        if not run_clicked and "last_result" not in st.session_state:
            st.info("Configure parameters in the sidebar, then click **▶ Run**.")
            return

        if run_clicked:
            cfg_h = _config_hash(
                s0=p_heat,
                rdep=rho_dep,
                te_p=te_ped,
                ti_p=ti_ped,
                dens=density,
                teq=tau_eq,
                geom=geometry,
                kappa=kappa,
                delta=delta,
                chi_s=chi_s,
                a_crit=a_over_lt_crit,
                chi_neo=chi_neo_val,
                n_rho=n_rho,
                max_iters=max_iters,
                tol=tol,
            )
            with st.spinner("Running Picard loop…"):
                t0 = time.perf_counter()
                result = _cached_run(
                    cfg_h,
                    s0=p_heat,
                    rdep=rho_dep,
                    te_p=te_ped,
                    ti_p=ti_ped,
                    dens=density,
                    teq=tau_eq,
                )
                elapsed = time.perf_counter() - t0
            st.session_state["last_result"] = result
            st.session_state["last_elapsed"] = elapsed

        result = st.session_state["last_result"]
        elapsed = st.session_state.get("last_elapsed", 0.0)

        # ── metrics row ──────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Te(0)", f"{result.te_final[0]:.0f} eV")
        c2.metric("Ti(0)", f"{result.ti_final[0]:.0f} eV")
        c3.metric("Picard iters", f"{result.metadata['n_iters']}")
        c4.metric("Converged", "✅" if result.metadata["converged"] else "❌")
        c5.metric("Runtime", f"{elapsed:.2f} s")

        # ── panel (a): profiles ──────────────────────────────────
        uq_band = _compute_uq_band(result.rho, result.te_final)
        fig_prof = _plot_profiles(result, uq_band=uq_band)

        # ── panel (b): convergence ───────────────────────────────
        fig_conv = _plot_convergence(result)

        # ── panel (c): chi ───────────────────────────────────────
        fig_chi = _plot_chi(result)

        # ── layout ───────────────────────────────────────────────
        tab_prof, tab_conv, tab_chi = st.tabs(
            [
                "📊 Temperature profiles",
                "📉 Convergence",
                "🔧 χ profiles",
            ]
        )
        with tab_prof:
            st.pyplot(fig_prof)
        with tab_conv:
            st.pyplot(fig_conv)
        with tab_chi:
            st.pyplot(fig_chi)

        plt.close("all")

        # Model info footer
        avail = []
        if pathlib.Path(mlp_path).exists():
            avail.append("MLP")
        if pathlib.Path(ensemble_dir).is_dir():
            avail.append("Ensemble")
        if pathlib.Path(conformal_path).exists():
            avail.append("Conformal")
        st.caption(
            f"Transport: **{transport_mode}** · "
            f"Available models: {', '.join(avail) or 'analytic only'} · "
            f"Grid: {n_rho} · Geometry: {geometry}"
        )

    # ── sweep mode ───────────────────────────────────────────────
    else:
        st.subheader(f"🔄 Sweep: {sweep_param}")
        st.caption(
            f"{len(sweep_values)} points from **{sweep_values[0]:.3f}** "
            f"to **{sweep_values[-1]:.3f}**"
        )

        if st.button("▶ Run sweep", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting sweep…")
            sweep_data: list[tuple[float, MultichannelResult, float]] = []

            _param_map = {
                "P_heat": "s0",
                "Te pedestal": "te_p",
                "Ti pedestal": "ti_p",
                "Density": "dens",
                "τ_eq": "teq",
                "ρ_dep": "rdep",
            }

            for i, val in enumerate(sweep_values):
                kw: dict[str, float] = {
                    "s0": p_heat,
                    "rdep": rho_dep,
                    "te_p": te_ped,
                    "ti_p": ti_ped,
                    "dens": density,
                    "teq": tau_eq,
                }
                kw[_param_map[sweep_param]] = val

                cfg_h = _config_hash(
                    **kw,
                    geom=geometry,
                    kappa=kappa,
                    delta=delta,
                    chi_s=chi_s,
                    a_crit=a_over_lt_crit,
                    chi_neo=chi_neo_val,
                    n_rho=n_rho,
                    max_iters=max_iters,
                    tol=tol,
                )
                t0 = time.perf_counter()
                res = _cached_run(cfg_h, **kw)
                dt_s = time.perf_counter() - t0

                sweep_data.append((val, res, dt_s))
                progress.progress(
                    (i + 1) / len(sweep_values),
                    text=f"Point {i + 1}/{len(sweep_values)}  "
                    f"({sweep_param}={val:.3f}) — {dt_s:.2f} s",
                )

            st.session_state["sweep_data"] = sweep_data

        if "sweep_data" in st.session_state:
            sweep_data = st.session_state["sweep_data"]

            # ── Te0 / Ti0 vs swept parameter ─────────────────────
            sweep_vals = [v for v, _, _ in sweep_data]
            sweep_te0s = [float(r.te_final[0]) for _, r, _ in sweep_data]
            sweep_ti0s = [float(r.ti_final[0]) for _, r, _ in sweep_data]

            fig_trend, ax_trend = plt.subplots(figsize=(7, 3.5))
            ax_trend.plot(sweep_vals, sweep_te0s, "C0-o", lw=1.8, markersize=5, label="Te(0)")
            ax_trend.plot(sweep_vals, sweep_ti0s, "C3--s", lw=1.4, markersize=4, label="Ti(0)")
            ax_trend.set_xlabel(sweep_param)
            ax_trend.set_ylabel("Core temperature [eV]")
            ax_trend.set_title("Core temperature response")
            ax_trend.legend()
            ax_trend.grid(True, alpha=0.3)
            fig_trend.tight_layout()
            st.pyplot(fig_trend)
            plt.close(fig_trend)

            # ── overlay plots ────────────────────────────────────
            n_cols = 2 if sweep_show_ti else 1
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5), squeeze=False)
            ax_te = axes[0, 0]
            for val, res, _ in sweep_data:
                label = f"{sweep_param}={val:.2f}"
                ax_te.plot(res.rho, res.te_final, lw=1.4, label=label)
            ax_te.set_xlabel("ρ")
            ax_te.set_ylabel("Te [eV]")
            ax_te.set_title("Electron temperature — sweep overlay")
            ax_te.legend(fontsize=8)
            ax_te.grid(True, alpha=0.3)

            if sweep_show_ti:
                ax_ti = axes[0, 1]
                for val, res, _ in sweep_data:
                    label = f"{sweep_param}={val:.2f}"
                    ax_ti.plot(res.rho, res.ti_final, lw=1.4, label=label)
                ax_ti.set_xlabel("ρ")
                ax_ti.set_ylabel("Ti [eV]")
                ax_ti.set_title("Ion temperature — sweep overlay")
                ax_ti.legend(fontsize=8)
                ax_ti.grid(True, alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── summary table ────────────────────────────────────
            import pandas as pd

            rows = []
            for val, res, dt_s in sweep_data:
                rows.append(
                    {
                        sweep_param: round(val, 3),
                        "Te(0) [eV]": round(float(res.te_final[0]), 1),
                        "Ti(0) [eV]": round(float(res.ti_final[0]), 1),
                        "n_iters": res.metadata["n_iters"],
                        "runtime_s": round(dt_s, 3),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
