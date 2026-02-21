"""Run the Picard iteration loop from a YAML config.

Usage
-----
    python -m scripts.run_picard --config configs/picard.yaml
    python -m scripts.run_picard --config configs/scenarios/mid_power.yaml

A *scenario* YAML may omit ``picard`` / ``solver.sub_steps`` keys — in
that case sensible defaults are used.

Produces (under ``<output_dir>/<run_id>/``)
--------
- ``result.npz``       — rho, Te, chi, residual & alpha histories
- ``metrics.json``     — convergence metadata + profile stats
- ``profiles.png``     — Te(rho) and chi(rho)
- ``convergence.png``  — residual history and relaxation schedule
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tokamak_transport_lab.integration.picard import run_picard


# ── defaults matching configs/picard.yaml ────────────────────────

_PICARD_DEFAULTS: dict[str, object] = {
    "max_iters": 50,
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

_TRANSPORT_DEFAULTS: dict[str, object] = {
    "chi_s": 1.0,
    "a_over_LTe_crit": 3.0,
    "alpha_s": 1.5,
    "chi_neo": 0.01,
}


def _get(cfg: dict, section: str, key: str, defaults: dict) -> object:
    """Get ``cfg[section][key]`` with fallback to *defaults*."""
    return cfg.get(section, {}).get(key, defaults[key])


# ── plotting helpers ─────────────────────────────────────────────


def _plot_profiles(result, out_path: pathlib.Path) -> None:
    """Save Te(rho) + chi(rho) side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax_te = axes[0]
    ax_te.plot(result.rho, result.te_final, "b-", linewidth=1.4)
    ax_te.set_xlabel(r"$\rho$")
    ax_te.set_ylabel(r"$T_e$ [eV]")
    ax_te.set_title("Temperature profile")
    ax_te.grid(True, alpha=0.3)

    ax_chi = axes[1]
    ax_chi.plot(result.rho, result.chi_profile, "r-", linewidth=1.4)
    ax_chi.set_xlabel(r"$\rho$")
    ax_chi.set_ylabel(r"$\chi$ [m$^2$/s]")
    ax_chi.set_title("Diffusivity profile")
    ax_chi.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_convergence(result, tol: float, out_path: pathlib.Path) -> None:
    """Save residual + alpha history side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax_res = axes[0]
    ax_res.semilogy(result.residual_history, "b-", linewidth=1.2)
    ax_res.axhline(tol, color="r", linestyle="--", label=f"tol={tol:.0e}")
    ax_res.set_xlabel("Picard iteration")
    ax_res.set_ylabel("Relative L2 residual")
    ax_res.set_title("Convergence history")
    ax_res.legend()
    ax_res.grid(True, alpha=0.3)

    ax_alp = axes[1]
    ax_alp.plot(result.alpha_history, "g-", linewidth=1.2)
    ax_alp.set_xlabel("Picard iteration")
    ax_alp.set_ylabel(r"$\alpha$ (under-relaxation)")
    ax_alp.set_title("Relaxation schedule")
    ax_alp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Picard iteration loop")
    parser.add_argument("--config", type=str, default="configs/picard.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Sub-folder name (default: timestamp)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Output directory: <base>/<run_id>/ ───────────────────────
    base_dir = pathlib.Path(args.output_dir or cfg.get("output_dir", "outputs/picard"))
    run_id = args.run_id or datetime.datetime.now(tz=datetime.UTC).strftime(
        "%Y%m%dT%H%M%S"
    )
    out_dir = base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Unpack config with defaults ──────────────────────────────
    n_rho = cfg.get("grid", {}).get("n_rho", 100)
    te_ped = cfg.get("bc", {}).get("Te_ped", 500.0)
    s0 = cfg.get("source", {}).get("S0", 1.0)
    rho_dep = cfg.get("source", {}).get("rho_dep", 0.3)
    sigma = cfg.get("source", {}).get("sigma", 0.1)

    dt = _get(cfg, "solver", "dt", _SOLVER_DEFAULTS)
    theta = _get(cfg, "solver", "theta", _SOLVER_DEFAULTS)
    sub_steps = _get(cfg, "solver", "sub_steps", _SOLVER_DEFAULTS)

    max_iters = _get(cfg, "picard", "max_iters", _PICARD_DEFAULTS)
    tol = _get(cfg, "picard", "tol", _PICARD_DEFAULTS)
    alpha0 = _get(cfg, "picard", "alpha0", _PICARD_DEFAULTS)
    alpha_min = _get(cfg, "picard", "alpha_min", _PICARD_DEFAULTS)
    alpha_max = _get(cfg, "picard", "alpha_max", _PICARD_DEFAULTS)
    div_patience = _get(cfg, "picard", "divergence_patience", _PICARD_DEFAULTS)

    transport_cfg = cfg.get("transport", {})
    transport_params = {
        "chi_s": transport_cfg.get("chi_s", _TRANSPORT_DEFAULTS["chi_s"]),
        "a_over_LTe_crit": transport_cfg.get(
            "a_over_LTe_crit", _TRANSPORT_DEFAULTS["a_over_LTe_crit"]
        ),
        "alpha_s": transport_cfg.get("alpha_s", _TRANSPORT_DEFAULTS["alpha_s"]),
        "chi_neo": transport_cfg.get("chi_neo", _TRANSPORT_DEFAULTS["chi_neo"]),
    }

    # ── Geometry function ────────────────────────────────────────
    geom_name = cfg.get("geometry", "circular")
    v_prime_fn = None
    if geom_name == "miller":
        from functools import partial

        from tokamak_transport_lab.geometry.miller import vprime_miller

        miller_cfg = cfg.get("miller", {})
        v_prime_fn = partial(
            vprime_miller,
            kappa=miller_cfg.get("kappa", 1.0),
            delta=miller_cfg.get("delta", 0.0),
        )

    # ── Transport model ──────────────────────────────────────────
    from tokamak_transport_lab.transport.stiffness import chi_total

    # ── Run Picard ───────────────────────────────────────────────
    print(f"Running Picard iteration loop  (config={args.config}) …")
    result = run_picard(
        n_rho=n_rho,
        t_ped=te_ped,
        s0=s0,
        rho_dep=rho_dep,
        sigma=sigma,
        v_prime_fn=v_prime_fn,
        transport_model=chi_total,
        transport_params=transport_params,
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

    # ── Print summary ────────────────────────────────────────────
    m = result.metadata
    print(f"\nPicard finished in {m['n_iters']} iterations ({m['wall_time_s']:.2f} s)")
    print(f"  converged      : {m['converged']}")
    print(f"  used_fallback  : {m['used_fallback']}")
    print(f"  final residual : {m['final_residual']:.2e}")
    print(f"  Te_core        : {result.te_final[0]:.1f} eV")
    print(f"  Te_ped         : {result.te_final[-1]:.1f} eV")

    # ── Save result.npz ──────────────────────────────────────────
    npz_path = out_dir / "result.npz"
    np.savez(
        npz_path,
        rho=result.rho,
        Te=result.te_final,
        chi=result.chi_profile,
        residual_history=np.array(result.residual_history),
        alpha_history=np.array(result.alpha_history),
    )
    print(f"\nresult.npz      → {npz_path}")

    # ── Save metrics.json ────────────────────────────────────────
    metrics = {
        **result.metadata,
        "config": args.config,
        "n_rho": n_rho,
        "Te_core_eV": float(result.te_final[0]),
        "Te_ped_eV": float(result.te_final[-1]),
        "chi_core": float(result.chi_profile[0]),
        "chi_edge": float(result.chi_profile[-1]),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"metrics.json    → {metrics_path}")

    # ── profiles.png ─────────────────────────────────────────────
    profiles_path = out_dir / "profiles.png"
    _plot_profiles(result, profiles_path)
    print(f"profiles.png    → {profiles_path}")

    # ── convergence.png ──────────────────────────────────────────
    conv_path = out_dir / "convergence.png"
    _plot_convergence(result, float(tol), conv_path)
    print(f"convergence.png → {conv_path}")


if __name__ == "__main__":
    main()
