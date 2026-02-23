"""Streamlit interactive demo for tokamak-transport-lab.

Launch with::

    streamlit run src/tokamak_transport_lab/app/demo.py

Requires the ``demo`` extra::

    pip install -e ".[demo]"

The app provides:
- Sidebar sliders for heating power, pedestal temperature, density, geometry.
- Live Te(ρ) and Ti(ρ) profile plots.
- Convergence trace (residual vs Picard iteration).
- Sweep mode: vary one parameter and overlay results.
"""

from __future__ import annotations


def main() -> None:  # noqa: C901 — single-page app, complexity is expected
    """Entry point — all streamlit imports are deferred here."""
    import time
    from functools import partial

    import numpy as np
    import streamlit as st

    from tokamak_transport_lab.integration.multichannel_picard import (
        run_picard_multichannel,
    )
    from tokamak_transport_lab.transport.stiffness import chi_total

    # ── page config ──────────────────────────────────────────────
    st.set_page_config(
        page_title="tokamak-transport-lab",
        page_icon="🔥",
        layout="wide",
    )
    st.title("🔥 tokamak-transport-lab — Interactive Demo")
    st.markdown(
        "Coupled *Te*–*Ti* 1-D transport solver with stiffness model "
        "and Picard iteration.  Drag the sliders and watch the profiles update."
    )

    # ── sidebar ──────────────────────────────────────────────────
    st.sidebar.header("Physics parameters")

    s0_e = st.sidebar.slider(
        "Heating power S₀ₑ",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Gaussian source amplitude for electrons.",
    )
    te_ped = st.sidebar.slider(
        "Te pedestal [eV]",
        min_value=100.0,
        max_value=3000.0,
        value=500.0,
        step=50.0,
    )
    ti_ped = st.sidebar.slider(
        "Ti pedestal [eV]",
        min_value=100.0,
        max_value=2500.0,
        value=400.0,
        step=50.0,
    )
    density = st.sidebar.slider(
        "Density n₀",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    tau_eq = st.sidebar.slider(
        "τ_eq (equilibration time)",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
    )

    st.sidebar.header("Geometry")
    geometry = st.sidebar.selectbox("Geometry", ["circular", "miller"])
    kappa = 1.0
    delta = 0.0
    if geometry == "miller":
        kappa = st.sidebar.slider("κ (elongation)", 1.0, 2.5, 1.7, 0.1)
        delta = st.sidebar.slider("δ (triangularity)", 0.0, 0.5, 0.33, 0.01)

    st.sidebar.header("Transport")
    chi_s = st.sidebar.slider("χ_s (stiffness coeff)", 0.1, 5.0, 1.0, 0.1)
    a_over_lt_crit = st.sidebar.slider("a/L_T crit", 1.0, 8.0, 3.0, 0.5)

    st.sidebar.header("Solver / Picard")
    n_rho = st.sidebar.slider("Grid points", 20, 150, 60, 10)
    max_iters = st.sidebar.slider("Max Picard iters", 10, 100, 40, 5)
    tol = st.sidebar.select_slider(
        "Convergence tol",
        options=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        value=1e-3,
    )

    # ── sweep mode ───────────────────────────────────────────────
    st.sidebar.header("🔄 Sweep mode")
    sweep_enabled = st.sidebar.checkbox("Enable parameter sweep")
    sweep_param = "S₀ₑ"
    sweep_values: list[float] = []
    if sweep_enabled:
        sweep_param = st.sidebar.selectbox(
            "Parameter to sweep",
            ["S₀ₑ", "Te pedestal", "Ti pedestal", "Density", "τ_eq"],
        )
        sweep_start = st.sidebar.number_input("Start", value=0.5)
        sweep_end = st.sidebar.number_input("End", value=5.0)
        sweep_steps = st.sidebar.slider("Steps", 3, 10, 5)
        sweep_values = list(np.linspace(sweep_start, sweep_end, sweep_steps))

    # ── helpers ──────────────────────────────────────────────────

    def _build_vprime(geom: str, kap: float, delt: float):  # noqa: ANN202
        if geom == "miller":
            from tokamak_transport_lab.geometry.miller import vprime_miller

            return partial(vprime_miller, kappa=kap, delta=delt)
        return None

    def _run(  # noqa: ANN202
        *,
        s0: float,
        te_p: float,
        ti_p: float,
        dens: float,
        teq: float,
    ):
        params = {
            "chi_s": chi_s,
            "a_over_LTe_crit": a_over_lt_crit,
            "alpha_s": 1.5,
            "chi_neo": 0.01,
        }
        return run_picard_multichannel(
            n_rho=n_rho,
            te_ped=te_p,
            ti_ped=ti_p,
            s0_e=s0,
            s0_i=0.0,
            density=dens,
            tau_eq=teq,
            v_prime_fn=_build_vprime(geometry, kappa, delta),
            transport_model_e=chi_total,
            transport_params_e=params,
            transport_model_i=chi_total,
            transport_params_i={**params, "chi_s": params["chi_s"] * 0.5},
            dt=1e-4,
            sub_steps=200,
            max_iters=max_iters,
            tol=tol,
            alpha0=0.4,
        )

    # ── single run ───────────────────────────────────────────────
    if not sweep_enabled:
        t0 = time.perf_counter()
        result = _run(s0=s0_e, te_p=te_ped, ti_p=ti_ped, dens=density, teq=tau_eq)
        elapsed = time.perf_counter() - t0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Te(0)", f"{result.te_final[0]:.0f} eV")
        col2.metric("Ti(0)", f"{result.ti_final[0]:.0f} eV")
        col3.metric("Picard iters", f"{result.metadata['n_iters']}")
        col4.metric("Runtime", f"{elapsed:.2f} s")

        # ── profile plot ─────────────────────────────────────────
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(result.rho, result.te_final, "C0-", linewidth=2, label="Te")
        ax1.plot(result.rho, result.ti_final, "C3--", linewidth=2, label="Ti")
        ax1.set_xlabel("ρ")
        ax1.set_ylabel("Temperature [eV]")
        ax1.set_title("Temperature profiles")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(result.residual_history, "k-", linewidth=1.5)
        ax2.axhline(tol, color="r", linestyle=":", label=f"tol = {tol:.0e}")
        ax2.set_xlabel("Picard iteration")
        ax2.set_ylabel("Relative L₂ residual")
        ax2.set_title("Convergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── sweep mode ───────────────────────────────────────────────
    else:
        import matplotlib.pyplot as plt

        st.subheader(f"Sweep: {sweep_param}")
        progress = st.progress(0)

        results = []
        for i, val in enumerate(sweep_values):
            kwargs = {
                "s0": s0_e,
                "te_p": te_ped,
                "ti_p": ti_ped,
                "dens": density,
                "teq": tau_eq,
            }
            if sweep_param == "S₀ₑ":
                kwargs["s0"] = val
            elif sweep_param == "Te pedestal":
                kwargs["te_p"] = val
            elif sweep_param == "Ti pedestal":
                kwargs["ti_p"] = val
            elif sweep_param == "Density":
                kwargs["dens"] = val
            elif sweep_param == "τ_eq":
                kwargs["teq"] = val

            res = _run(**kwargs)
            results.append((val, res))
            progress.progress((i + 1) / len(sweep_values))

        # overlay plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for val, res in results:
            label = f"{sweep_param}={val:.2f}"
            ax1.plot(res.rho, res.te_final, linewidth=1.4, label=label)
            ax2.plot(res.rho, res.ti_final, linewidth=1.4, label=label)

        ax1.set_xlabel("ρ")
        ax1.set_ylabel("Te [eV]")
        ax1.set_title("Electron temperature sweep")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("ρ")
        ax2.set_ylabel("Ti [eV]")
        ax2.set_title("Ion temperature sweep")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # summary table
        import pandas as pd

        rows = []
        for val, res in results:
            rows.append(
                {
                    sweep_param: round(val, 3),
                    "Te(0) [eV]": round(float(res.te_final[0]), 1),
                    "Ti(0) [eV]": round(float(res.ti_final[0]), 1),
                    "Picard iters": res.metadata["n_iters"],
                    "Converged": res.metadata["converged"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
