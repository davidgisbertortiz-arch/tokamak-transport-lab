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
- ``scenarios_compare.png``     — Te + Ti profiles side-by-side
- ``comparison_convergence.png``— residual histories overlaid
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tokamak_transport_lab.integration.multichannel_picard import (
    MultichannelResult,
)

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
    ax.set_ylabel("Source-scaled PDE residual (max channel)")
    ax.set_title("Convergence comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── run one scenario ─────────────────────────────────────────────


def run_one_scenario(cfg: dict) -> MultichannelResult:
    """Parse a scenario config dict and run the multichannel Picard loop."""
    from tokamak_transport_lab.integration.config import run_config

    return run_config(cfg)


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
        raise SystemExit(f"No YAML files found in {scenarios_dir}")

    results: dict[str, MultichannelResult] = {}
    summary: dict[str, dict] = {}

    for yaml_path in yaml_files:
        name = yaml_path.stem
        print(f"\n{'=' * 60}")
        print(f"Running scenario: {name}  ({yaml_path})")
        print(f"{'=' * 60}")

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
            "final_residual": m["final_residual"],
            "runtime_s": round(m["wall_time_s"], 3),
            "Te0": round(float(result.te_final[0]), 1),
            "Ti0": round(float(result.ti_final[0]), 1),
            "Te_ped_eV": round(float(result.te_final[-1]), 1),
            "Ti_ped_eV": round(float(result.ti_final[-1]), 1),
        }

    # Global outputs
    summary_path = out_base / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary → {summary_path}")

    _plot_comparison_profiles(results, out_base / "scenarios_compare.png")
    print(f"Profiles plot → {out_base / 'scenarios_compare.png'}")

    _plot_comparison_convergence(results, out_base / "comparison_convergence.png")
    print(f"Convergence plot → {out_base / 'comparison_convergence.png'}")

    if not all(r.metadata["converged"] for r in results.values()):
        raise SystemExit("At least one scenario did not converge; inspect saved metrics.")
    print("\nAll scenarios converged.")


if __name__ == "__main__":
    main()
