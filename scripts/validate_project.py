"""Generate numerical, ML and uncertainty evidence from executable benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml

from tokamak_transport_lab.integration.config import run_config
from tokamak_transport_lab.integration.picard import run_picard
from tokamak_transport_lab.solver.analytic import steady_state_reference
from tokamak_transport_lab.solver.crank_nicolson import solve
from tokamak_transport_lab.solver.steady import stationary_solve
from tokamak_transport_lab.surrogate.bundle import TransportBundle
from tokamak_transport_lab.surrogate.profile_uq import family_config
from tokamak_transport_lab.surrogate.training import load_data, regression_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/validation")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = []
    for n in [31, 61, 121, 241]:
        rho = np.linspace(0, 1, n)
        source = np.exp(-((rho - 0.3) ** 2) / 0.02)
        numerical = stationary_solve(rho, [np.ones(n)], rho, [source], [0.0])[0]
        ref = steady_state_reference(rho, 1.0, 1.0, 0.3, 0.1, 0.0, n_quad=60001)
        grid.append(
            {
                "n_rho": n,
                "relative_rise_error": float(
                    np.linalg.norm(numerical - ref) / np.linalg.norm(ref)
                ),
            }
        )
    for previous, current in itertools.pairwise(grid):
        current["observed_order"] = float(
            np.log(previous["relative_rise_error"] / current["relative_rise_error"])
            / np.log((current["n_rho"] - 1) / (previous["n_rho"] - 1))
        )
    assert all(1.9 < r["observed_order"] < 2.1 for r in grid[1:])
    transient = solve()
    assert transient["converged"]
    starts = []
    for temperature in [500.0, 1500.0, 10000.0]:
        r = run_picard(n_rho=40, te_init=np.full(40, temperature), tol=1e-7)
        assert r.metadata["converged"]
        starts.append(r.te_final)
    initial_spread = float(np.max(np.ptp(starts, axis=0)))
    assert initial_spread < 1e-6
    scenarios = {}
    for path in sorted(Path("configs/scenarios").glob("*.yaml")):
        result = run_config(yaml.safe_load(path.read_text()))
        assert result.metadata["converged"]
        scenarios[path.stem] = result.metadata
    legacy = {}
    for path in sorted(Path("tests/fixtures/legacy_scenarios").glob("*.yaml")):
        result = run_config(yaml.safe_load(path.read_text()))
        assert result.metadata["converged"]
        legacy[path.stem] = result.metadata
    ensemble = TransportBundle.load("outputs/ensemble/model.pt")
    mlp = TransportBundle.load("outputs/surrogate/model.pt")
    x, y, _ = load_data("data/transport_test.npz")
    epred = ensemble.predict_chi(x)
    ml_metrics = {
        "ensemble": regression_metrics(y, epred),
        "mlp": regression_metrics(y, mlp.predict_chi(x)),
    }
    cfg = family_config()
    analytic = run_config(cfg)
    cfg["model"] = {"type": "ensemble"}
    learned = run_config(cfg, bundle=ensemble)
    assert learned.metadata["converged"] and analytic.metadata["converged"]
    profile_error = float(
        max(
            np.max(abs(learned.te_final - analytic.te_final)),
            np.max(abs(learned.ti_final - analytic.ti_final)),
        )
    )
    uq = json.loads(Path("outputs/profile_uq/calibration.json").read_text())
    local_uq = json.loads(Path("outputs/conformal/summary.json").read_text())
    assert uq["artifact_sha256"] == ensemble.fingerprint == local_uq["artifact_sha256"]
    profile_scores = np.load("outputs/profile_uq/test_profiles.npz")["scores"]
    source_files = sorted(
        [
            *Path("src").rglob("*.py"),
            *Path("scripts").glob("*.py"),
            *Path("configs").rglob("*.yaml"),
        ]
    )
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    summary = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": digest.hexdigest(),
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "grid_refinement": grid,
        "default_transient_steady_residual": transient["steady_residual"],
        "initial_condition_spread_eV": initial_spread,
        "scenarios": scenarios,
        "legacy_scenarios": legacy,
        "ml_test_metrics": ml_metrics,
        "ensemble_sha256": ensemble.fingerprint,
        "mlp_sha256": mlp.fingerprint,
        "reference_profile_max_error_eV": profile_error,
        "local_conformal": local_uq,
        "profile_conformal": uq,
        "scope": "Synthetic closure emulation and numerical verification; no experimental plasma validation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        out / "reference_profiles.npz",
        rho=analytic.rho,
        Te_analytic=analytic.te_final,
        Ti_analytic=analytic.ti_final,
        Te_ensemble=learned.te_final,
        Ti_ensemble=learned.ti_final,
    )
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axs[0, 0]
    ax.loglog(
        [r["n_rho"] - 1 for r in grid],
        [r["relative_rise_error"] for r in grid],
        "o-",
        color="#2563eb",
    )
    ax.set(
        xlabel="Radial intervals",
        ylabel="Relative temperature-rise error",
        title="Diffusion: second-order spatial convergence",
    )
    ax = axs[0, 1]
    ax.loglog(y, epred, ".", alpha=0.2, markersize=2, color="#2563eb")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color="#d97706", ls="--")
    ax.set(
        xlabel="Synthetic reference diffusivity",
        ylabel="Ensemble prediction",
        title=f"Held-out local test: R² = {ml_metrics['ensemble']['r2']:.5f}",
    )
    ax = axs[1, 0]
    ax.plot(analytic.rho, analytic.te_final, color="#2563eb", label="Te reference")
    ax.plot(analytic.rho, learned.te_final, "--", color="#16a34a", label="Te ensemble")
    ax.plot(analytic.rho, analytic.ti_final, color="#dc2626", label="Ti reference")
    ax.plot(analytic.rho, learned.ti_final, "--", color="#d97706", label="Ti ensemble")
    ax.set(
        xlabel="Normalized radius",
        ylabel="Temperature [eV]",
        title=f"Coupled profile error: {profile_error:.2f} eV maximum",
    )
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    scores = np.sort(profile_scores)
    ax.step(
        scores,
        np.arange(1, len(scores) + 1) / len(scores),
        where="post",
        color="#2563eb",
        label="Independent test cases",
    )
    ax.axvline(
        uq["q_hat_eV"],
        color="#d97706",
        ls="--",
        label=f"Calibrated radius: {uq['q_hat_eV']:.2f} eV",
    )
    ax.axhline(0.9, color="grey", ls=":")
    ax.set(
        xlabel="Maximum Te/Ti error over all radial nodes [eV]",
        ylabel="Fraction of scenarios covered",
        title=f"Simultaneous profile coverage: {uq['test_simultaneous_coverage']:.1%} ({len(scores)} cases)",
    )
    ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=0.2)
    fig.savefig(out / "validation.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "grid_refinement": grid,
                "profile_max_error_eV": profile_error,
                "ml": ml_metrics,
                "output": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
