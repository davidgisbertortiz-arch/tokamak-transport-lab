"""Interactive, reproducible experiments with the shared stationary solver.

Run: streamlit run src/tokamak_transport_lab/app/demo.py
"""

from __future__ import annotations


def main():
    import hashlib
    import io
    import json
    from copy import deepcopy
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st
    import yaml

    from tokamak_transport_lab.integration.config import ensemble_profiles, run_config
    from tokamak_transport_lab.surrogate.profile_uq import FAMILY, family_config, profile_interval

    st.set_page_config(page_title="Tokamak Transport Lab", page_icon="🌀", layout="wide")
    st.title("Tokamak Transport Lab")
    st.caption(
        "A synthetic radial heat-transport experiment · coupled electron and ion temperatures"
    )
    root = Path(__file__).resolve().parents[3]
    preset = st.sidebar.selectbox(
        "Experiment",
        ["Profile benchmark", "Low heating", "Medium heating", "High heating", "Custom"],
    )
    if preset in ["Low heating", "Medium heating", "High heating"]:
        name = {
            "Low heating": "low_power",
            "Medium heating": "mid_power",
            "High heating": "high_power",
        }[preset]
        cfg = yaml.safe_load((root / "configs/scenarios" / f"{name}.yaml").read_text())
    else:
        cfg = family_config()
    if preset == "Profile benchmark":
        source = st.sidebar.slider(
            "Electron heating amplitude [eV / normalized time]", 300.0, 1000.0, 650.0, 25.0
        )
        ped = st.sidebar.slider("Electron pedestal [eV]", 300.0, 700.0, 500.0, 25.0)
        cfg = family_config(source, ped)
        st.sidebar.caption(
            "Fixed circular geometry and closure parameters. Ti pedestal = 0.8 × Te pedestal."
        )
    elif preset == "Custom":
        cfg["source"]["S0_e"] = st.sidebar.slider(
            "Electron heating amplitude [eV / normalized time]", 1.0, 5000.0, 650.0, 1.0
        )
        cfg["source"]["rho_dep"] = st.sidebar.slider("Deposition radius", 0.0, 0.8, 0.3, 0.05)
        cfg["bc"]["Te_ped"] = st.sidebar.slider(
            "Electron pedestal [eV]", 100.0, 1500.0, 500.0, 25.0
        )
        cfg["bc"]["Ti_ped"] = st.sidebar.slider("Ion pedestal [eV]", 100.0, 1500.0, 400.0, 25.0)
        cfg["coupling"]["density"] = st.sidebar.slider(
            "Normalized exchange multiplier", 0.1, 3.0, 1.0, 0.1
        )
        cfg["coupling"]["tau_eq"] = st.sidebar.slider(
            "Exchange time [normalized]", 0.1, 5.0, 1.0, 0.1
        )
        cfg["grid"]["n_rho"] = st.sidebar.slider("Radial grid points", 20, 100, 40, 10)
        for channel, label in [("transport_e", "Electron"), ("transport_i", "Ion")]:
            with st.sidebar.expander(label + " transport"):
                cfg[channel]["chi_s"] = st.slider(
                    label + " stiffness", 0.1, 5.0, cfg[channel]["chi_s"], 0.1
                )
                cfg[channel]["a_over_LTe_crit"] = st.slider(
                    label + " critical gradient", 1.0, 6.0, cfg[channel]["a_over_LTe_crit"], 0.1
                )
                cfg[channel]["chi_neo"] = st.number_input(
                    label + " diffusivity floor",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    format="%.3f",
                )
    mode = st.sidebar.selectbox("Transport model", ["analytic", "mlp", "ensemble"])
    model_path = "outputs/surrogate/model.pt" if mode == "mlp" else "outputs/ensemble/model.pt"
    if mode != "analytic":
        model_path = st.sidebar.text_input("Trained model artifact", value=model_path)
    cfg["model"] = {"type": mode, "path": model_path}
    show_range = (
        st.sidebar.checkbox("Propagate ensemble members", value=False, disabled=mode != "ensemble")
        and mode == "ensemble"
    )
    show_calibrated = st.sidebar.checkbox(
        "Calibrated profile interval",
        value=False,
        disabled=mode != "ensemble" or preset != "Profile benchmark",
    )
    calibration_path = root / "outputs/profile_uq/calibration.json"
    sweep = st.sidebar.checkbox("Sweep electron heating")
    n_sweep = st.sidebar.slider("Sweep points", 3, 8, 4) if sweep else 1
    st.sidebar.caption(
        "Sources and diffusivities are normalized. These experiments do not predict a tokamak device or heating power in MW."
    )

    @st.cache_data(show_spinner=False)
    def execute(config_json, artifact_digest, with_members):
        config = json.loads(config_json)
        result = run_config(config)
        band = ensemble_profiles(config) if with_members and result.metadata["converged"] else None
        return result, band

    if st.button("Run experiment", type="primary"):
        try:
            digest = (
                hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
                if mode != "analytic"
                else "analytic"
            )
            configs = [deepcopy(cfg)]
            if sweep:
                values = (
                    np.linspace(FAMILY["source_e"][0], FAMILY["source_e"][1], n_sweep)
                    if preset == "Profile benchmark"
                    else np.linspace(
                        0.5 * cfg["source"]["S0_e"], 1.5 * cfg["source"]["S0_e"], n_sweep
                    )
                )
                configs = []
                for value in values:
                    item = deepcopy(cfg)
                    item["source"]["S0_e"] = float(value)
                    configs.append(item)
            runs = []
            progress = st.progress(0.0, text="Solving the stationary transport equations…")
            for i, config in enumerate(configs):
                result, band = execute(json.dumps(config, sort_keys=True), digest, show_range)
                calibrated = None
                if show_calibrated and mode == "ensemble" and preset == "Profile benchmark":
                    record = json.loads(calibration_path.read_text())
                    if record["artifact_sha256"] != digest or record["family"] != FAMILY:
                        raise ValueError(
                            "Profile calibration does not match this model or experiment family. Recalibrate it."
                        )
                    calibrated = profile_interval(result, record["q_hat_eV"])
                    calibrated["calibration"] = record
                runs.append((config, result, band, calibrated))
                progress.progress((i + 1) / len(configs))
            st.session_state["experiment"] = {
                "runs": runs,
                "submitted_config": deepcopy(cfg),
                "preset": preset,
                "options": [show_range, show_calibrated, sweep, n_sweep],
            }
            progress.empty()
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            st.error(f"Experiment could not be completed: {exc}")
            return
    if "experiment" not in st.session_state:
        st.info(
            "Choose an experiment and model, then run it. ML models are created by the training commands in the README."
        )
        return
    snapshot = st.session_state["experiment"]
    if (
        snapshot["submitted_config"] != cfg
        or snapshot["options"] != [show_range, show_calibrated, sweep, n_sweep]
        or snapshot["preset"] != preset
    ):
        st.info("Showing the previous experiment. Run again to apply the current controls.")
    runs = snapshot["runs"]
    config, result, band, calibrated = runs[-1]
    meta = result.metadata
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Electron core temperature", f"{result.te_final[0]:.1f} eV")
    c2.metric("Ion core temperature", f"{result.ti_final[0]:.1f} eV")
    c3.metric("PDE residual", f"{meta['final_residual']:.2e}")
    c4.metric("Converged", "Yes" if meta["converged"] else "No")
    if not all(item[1].metadata["converged"] for item in runs):
        st.error("At least one solve did not converge. These profiles are incomplete iterates.")
    if any(item[1].metadata["out_of_domain_fraction"] > 0 for item in runs):
        st.warning(
            "Some final closure inputs are outside the training range. The surrogate is extrapolating."
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for config, item, member_band, cal_band in runs:
        label = f"S0={config['source']['S0_e']:.0f}" if len(runs) > 1 else config["model"]["type"]
        axes[0].plot(item.rho, item.te_final, label="Te · " + label)
        axes[0].plot(item.rho, item.ti_final, ls="--", label="Ti · " + label)
        if member_band is not None:
            axes[0].fill_between(
                item.rho,
                member_band["te_lower"],
                member_band["te_upper"],
                alpha=0.18,
                label="Te member range",
            )
            axes[0].fill_between(
                item.rho,
                member_band["ti_lower"],
                member_band["ti_upper"],
                alpha=0.12,
                label="Ti member range",
            )
        if cal_band is not None:
            for channel in ("te", "ti"):
                axes[0].fill_between(
                    item.rho,
                    cal_band[channel + "_lower"],
                    cal_band[channel + "_upper"],
                    alpha=0.13,
                    color="grey",
                    label=channel.upper() + " calibrated interval",
                )
        axes[1].semilogy(item.residual_history, label=label)
        axes[2].plot(item.rho, item.chi_e_profile, label="χe · " + label)
        axes[2].plot(item.rho, item.chi_i_profile, ls="--", label="χi · " + label)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set(
        xlabel="Normalized radius ρ", ylabel="Temperature [eV]", title="Temperature profiles"
    )
    axes[1].set(
        xlabel="Nonlinear iteration",
        ylabel="Source-scaled PDE residual",
        title="Stationary balance",
    )
    axes[2].set(
        xlabel="Normalized radius ρ",
        ylabel="Normalized diffusivity",
        title="Transport used by the solver",
    )
    st.pyplot(fig)
    plt.close(fig)
    if band is not None:
        st.caption(
            "Member ranges show disagreement after solving each learned closure. They have no stated coverage probability."
        )
    if calibrated is not None:
        record = calibrated["calibration"]
        st.caption(
            f"Profile calibration: nominal {1 - record['alpha']:.0%}; independent test coverage {record['test_simultaneous_coverage']:.1%} over {record['n_test']} scenarios. Coverage is marginal over the stated random source/pedestal experiment, simultaneously for both profiles on this grid. It is not a guarantee for a selected parameter value or a real plasma."
        )
    st.caption(
        f"Displayed model: {meta['transport_model']} · {meta['n_iters']} iterations · {meta['wall_time_s']:.2f} s · circular geometry"
    )
    st.dataframe(
        [
            {
                "S0_e": cfg_i["source"]["S0_e"],
                "Te0_eV": r.te_final[0],
                "Ti0_eV": r.ti_final[0],
                "converged": r.metadata["converged"],
                "PDE_residual": r.metadata["final_residual"],
            }
            for cfg_i, r, _, _ in runs
        ]
    )
    with st.expander("Reproduce this experiment"):
        st.json({"config": config, "metrics": meta})
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        rho=result.rho,
        Te=result.te_final,
        Ti=result.ti_final,
        chi_e=result.chi_e_profile,
        chi_i=result.chi_i_profile,
    )
    st.download_button(
        "Download displayed profiles", buffer.getvalue(), file_name="transport_profiles.npz"
    )
    st.download_button(
        "Download configuration and metrics",
        json.dumps({"config": config, "metrics": meta}, indent=2),
        file_name="experiment.json",
    )


if __name__ == "__main__":
    main()
