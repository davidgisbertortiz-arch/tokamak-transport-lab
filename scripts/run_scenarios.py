"""Run all multichannel scenarios and produce comparison outputs.

Usage
-----
    python -m scripts.run_scenarios
    python -m scripts.run_scenarios --scenarios-dir configs/scenarios
    python -m scripts.run_scenarios --output-dir outputs/scenarios

Produces (under ``<output_dir>/``)
--------
- ``<scenario>/result.npz``     — per-scenario rho, Te, Ti, chi_e, chi_i, …
- ``<scenario>/metrics.json``   — convergence metadata + profile stats
- ``summary.json``              — comparison table for all scenarios
- ``comparison_profiles.png``   — Te + Ti profiles side-by-side
- ``comparison_convergence.png``— residual histories overlaid
"""

from __future__ import annotations

import argparse
import json
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tokamak_transport_lab.integration.multichannel_picard import (
    MultichannelResult,
    run_picard_multichannel,
)
from tokamak_transport_lab.transport.ion_transport import make_ion_transport
from tokamak_transport_lab.transport.stiffness import chi_total

# ── defaults ─────────────────────────────────────────────────────

_PICARD_DEFAULTS: dict[str, object] = {
    "max_iters": 60,
    "tol": 1e-4,
    "alpha0": 0.5,
    "alpha_min": 0.05,
    "alpha_max": 1.0,
    "divergence_patience": 5,
}

_SOLVER_DEFAULTS: dict[str, object] = {
    "dt": 1e-4,
    "theta": 0.5,
    "sub_steps": 500,
}

_TRANSPORT_DEFAULTS: dict[str, float] = {
    "chi_s": 1.0,
    "a_over_LTe_crit": 3.0,
    "alpha_s": 1.5,
    "chi_neo": 0.01,
}


def _get(cfg: dict, section: str, key: str, defaults: dict) -> object:
    return cfg.get(section, {}).get(key, defaults[key])


def _transport_params(cfg: dict, section: str) -> dict[str, float]:
    """Extract transport parameters from a config section."""
    tcfg = cfg.get(section, cfg.get("transport", {}))
    return {
        "chi_s": tcfg.get("chi_s", _TRANSPORT_DEFAULTS["chi_s"]),
        "a_over_LTe_crit": tcfg.get(
            "a_over_LTe_crit", _TRANSPORT_DEFAULTS["a_over_LTe_crit"]
        ),
        "alpha_s": tcfg.get("alpha_s", _TRANSPORT_DEFAULTS["alpha_s"]),
        "chi_neo": tcfg.get("chi_neo", _TRANSPORT_DEFAULTS["chi_neo"]),
    }


# ── plotting ─────────────────────────────────────────────────────


def _plot_comparison_profiles(
    results: dict[str, MultichannelResult],
    out_path: pathlib.Path,
) -> None:
    """Plot Te and Ti profiles for all scenarios side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_te, ax_ti = axes
    for name, res in results.items():
        ax_te.plot(res.rho, res.te_final, linewidth=1.4, label=name)
        ax_ti.plot(res.rho, res.ti_final, linewidth=1.4, label=name)

    ax_te.set_xlabel(r"$\rho$")
    ax_te.set_ylabel(r"$T_e$ [eV]")
    ax_te.set_title("Electron temperature")
    ax_te.legend()
    ax_te.grid(True, alpha=0.3)

    ax_ti.set_xlabel(r"$\rho$")
    ax_ti.set_ylabel(r"$T_i$ [eV]")
    ax_ti.set_title("Ion temperature")
    ax_ti.legend()
    ax_ti.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_comparison_convergence(
    results: dict[str, MultichannelResult],
    out_path: pathlib.Path,
) -> None:
    """Plot residual histories for all scenarios overlaid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        ax.semilogy(res.residual_history, linewidth=1.2, label=name)

    ax.set_xlabel("Picard iteration")
    ax.set_ylabel("Relative L2 residual (max of Te, Ti)")
    ax.set_title("Convergence comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── run one scenario ─────────────────────────────────────────────


def run_one_scenario(cfg: dict) -> MultichannelResult:
    """Parse a scenario config dict and run the multichannel Picard loop."""
    n_rho = cfg.get("grid", {}).get("n_rho", 100)

    source_cfg = cfg.get("source", {})
    s0_e = source_cfg.get("S0_e", source_cfg.get("S0", 1.0))
    s0_i = source_cfg.get("S0_i", 0.0)
    rho_dep = source_cfg.get("rho_dep", 0.3)
    sigma = source_cfg.get("sigma", 0.1)

    bc_cfg = cfg.get("bc", {})
    te_ped = bc_cfg.get("Te_ped", 500.0)
    ti_ped = bc_cfg.get("Ti_ped", te_ped * 0.8)

    coupling_cfg = cfg.get("coupling", {})
    density = coupling_cfg.get("density", 1.0)
    tau_eq = coupling_cfg.get("tau_eq", 0.01)

    dt = _get(cfg, "solver", "dt", _SOLVER_DEFAULTS)
    theta = _get(cfg, "solver", "theta", _SOLVER_DEFAULTS)
    sub_steps = _get(cfg, "solver", "sub_steps", _SOLVER_DEFAULTS)

    max_iters = _get(cfg, "picard", "max_iters", _PICARD_DEFAULTS)
    tol = _get(cfg, "picard", "tol", _PICARD_DEFAULTS)
    alpha0 = _get(cfg, "picard", "alpha0", _PICARD_DEFAULTS)
    alpha_min = _get(cfg, "picard", "alpha_min", _PICARD_DEFAULTS)
    alpha_max = _get(cfg, "picard", "alpha_max", _PICARD_DEFAULTS)
    div_patience = _get(cfg, "picard", "divergence_patience", _PICARD_DEFAULTS)

    params_e = _transport_params(cfg, "transport_e")

    # Ion transport: factory dispatches on ti_transport_mode
    ti_mode = cfg.get("ti_transport_mode", "stiffness")
    qi_factor = cfg.get("qi_factor", 1.0)
    params_i_raw = _transport_params(cfg, "transport_i")
    ion_model, ion_params = make_ion_transport(
        mode=ti_mode,
        electron_model=chi_total,
        electron_params=params_e,
        qi_factor=qi_factor,
        ion_params=params_i_raw,
    )

    # Geometry
    v_prime_fn = None
    geom_name = cfg.get("geometry", "circular")
    if geom_name == "miller":
        from tokamak_transport_lab.geometry.miller import vprime_miller

        miller_cfg = cfg.get("miller", {})
        v_prime_fn = partial(
            vprime_miller,
            kappa=miller_cfg.get("kappa", 1.0),
            delta=miller_cfg.get("delta", 0.0),
        )

    return run_picard_multichannel(
        n_rho=n_rho,
        te_ped=te_ped,
        ti_ped=ti_ped,
        s0_e=s0_e,
        s0_i=s0_i,
        rho_dep=rho_dep,
        sigma=sigma,
        density=density,
        tau_eq=tau_eq,
        v_prime_fn=v_prime_fn,
        transport_model_e=chi_total,
        transport_params_e=params_e,
        transport_model_i=ion_model,
        transport_params_i=ion_params,
        dt=dt,
        theta=theta,
        sub_steps=sub_steps,
        max_iters=max_iters,
        tol=tol,
        alpha0=alpha0,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        divergence_patience=div_patience,
    )


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all multichannel scenarios")
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default="configs/scenarios",
        help="Directory containing scenario YAML files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/scenarios",
    )
    args = parser.parse_args()

    scenarios_dir = pathlib.Path(args.scenarios_dir)
    out_base = pathlib.Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(scenarios_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {scenarios_dir}")
        return

    results: dict[str, MultichannelResult] = {}
    summary: dict[str, dict] = {}

    for yaml_path in yaml_files:
        name = yaml_path.stem
        print(f"\n{'='*60}")
        print(f"Running scenario: {name}  ({yaml_path})")
        print(f"{'='*60}")

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        result = run_one_scenario(cfg)
        results[name] = result

        m = result.metadata
        print(f"  converged : {m['converged']}  ({m['n_iters']} iters, {m['wall_time_s']:.2f} s)")
        print(f"  Te_core   : {result.te_final[0]:.1f} eV")
        print(f"  Ti_core   : {result.ti_final[0]:.1f} eV")
        print(f"  Te_ped    : {result.te_final[-1]:.1f} eV")
        print(f"  Ti_ped    : {result.ti_final[-1]:.1f} eV")
        print(f"  Ti < Te   : {bool((result.ti_final <= result.te_final + 1.0).all())}")

        # Save per-scenario outputs
        sc_dir = out_base / name
        sc_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            sc_dir / "result.npz",
            rho=result.rho,
            Te=result.te_final,
            Ti=result.ti_final,
            chi_e=result.chi_e_profile,
            chi_i=result.chi_i_profile,
            residual_history=np.array(result.residual_history),
            alpha_history=np.array(result.alpha_history),
        )
        (sc_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")

        summary[name] = {
            "converged": m["converged"],
            "n_iters": m["n_iters"],
            "wall_time_s": round(m["wall_time_s"], 3),
            "final_residual": m["final_residual"],
            "Te_core_eV": round(float(result.te_final[0]), 1),
            "Ti_core_eV": round(float(result.ti_final[0]), 1),
            "Te_ped_eV": round(float(result.te_final[-1]), 1),
            "Ti_ped_eV": round(float(result.ti_final[-1]), 1),
        }

    # Global outputs
    summary_path = out_base / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary → {summary_path}")

    _plot_comparison_profiles(results, out_base / "comparison_profiles.png")
    print(f"Profiles plot → {out_base / 'comparison_profiles.png'}")

    _plot_comparison_convergence(results, out_base / "comparison_convergence.png")
    print(f"Convergence plot → {out_base / 'comparison_convergence.png'}")

    print("\nAll scenarios complete.")


if __name__ == "__main__":
    main()
