"""Run the Picard iteration loop.

Usage
-----
    python -m scripts.run_picard --config configs/picard.yaml

Produces
--------
- ``outputs/picard/picard_result.npz`` — rho, Te, chi, residuals
- ``outputs/picard/picard_metadata.json`` — convergence info
- ``outputs/picard/convergence.png`` — residual & alpha history
"""

from __future__ import annotations

import argparse
import json
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tokamak_transport_lab.integration.picard import run_picard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Picard iteration loop")
    parser.add_argument("--config", type=str, default="configs/picard.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    picard_cfg = cfg["picard"]
    grid_cfg = cfg["grid"]
    solver_cfg = cfg["solver"]
    source_cfg = cfg["source"]
    bc_cfg = cfg["bc"]
    transport_cfg = cfg["transport"]
    geom_name = cfg.get("geometry", "circular")
    out_dir = pathlib.Path(args.output_dir or cfg.get("output_dir", "outputs/picard"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Geometry function ────────────────────────────────────────
    v_prime_fn = None
    if geom_name == "miller":
        from tokamak_transport_lab.geometry.miller import vprime_miller

        miller_cfg = cfg.get("miller", {})
        v_prime_fn = partial(
            vprime_miller,
            kappa=miller_cfg.get("kappa", 1.0),
            delta=miller_cfg.get("delta", 0.0),
        )

    # ── Transport model ──────────────────────────────────────────
    from tokamak_transport_lab.transport.stiffness import chi_total

    transport_params = {
        "chi_s": transport_cfg.get("chi_s", 1.0),
        "a_over_LTe_crit": transport_cfg.get("a_over_LTe_crit", 3.0),
        "alpha_s": transport_cfg.get("alpha_s", 1.5),
        "chi_neo": transport_cfg.get("chi_neo", 0.01),
    }

    # ── Run Picard ───────────────────────────────────────────────
    print("Running Picard iteration loop ...")
    result = run_picard(
        n_rho=grid_cfg["n_rho"],
        t_ped=bc_cfg["Te_ped"],
        s0=source_cfg["S0"],
        rho_dep=source_cfg["rho_dep"],
        sigma=source_cfg["sigma"],
        v_prime_fn=v_prime_fn,
        transport_model=chi_total,
        transport_params=transport_params,
        dt=solver_cfg["dt"],
        theta=solver_cfg["theta"],
        sub_steps=solver_cfg["sub_steps"],
        max_iters=picard_cfg["max_iters"],
        tol=picard_cfg["tol"],
        alpha0=picard_cfg["alpha0"],
        alpha_min=picard_cfg["alpha_min"],
        alpha_max=picard_cfg["alpha_max"],
        divergence_patience=picard_cfg["divergence_patience"],
    )

    # ── Print summary ────────────────────────────────────────────
    m = result.metadata
    print(f"\nPicard finished in {m['n_iters']} iterations ({m['wall_time_s']:.2f} s)")
    print(f"  converged:     {m['converged']}")
    print(f"  used_fallback: {m['used_fallback']}")
    print(f"  final residual: {m['final_residual']:.2e}")

    # ── Save NPZ ─────────────────────────────────────────────────
    npz_path = out_dir / "picard_result.npz"
    np.savez(
        npz_path,
        rho=result.rho,
        Te=result.te_final,
        chi=result.chi_profile,
        residual_history=np.array(result.residual_history),
        alpha_history=np.array(result.alpha_history),
    )
    print(f"Result   → {npz_path}")

    # ── Save metadata JSON ───────────────────────────────────────
    meta_path = out_dir / "picard_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(result.metadata, f, indent=2)
    print(f"Metadata → {meta_path}")

    # ── Convergence plot ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax_res = axes[0]
    ax_res.semilogy(result.residual_history, "b-", linewidth=1.2)
    ax_res.axhline(picard_cfg["tol"], color="r", linestyle="--", label=f"tol={picard_cfg['tol']}")
    ax_res.set_xlabel("Picard iteration")
    ax_res.set_ylabel("Relative L2 residual")
    ax_res.set_title("Convergence history")
    ax_res.legend()
    ax_res.grid(True, alpha=0.3)

    ax_alp = axes[1]
    ax_alp.plot(result.alpha_history, "g-", linewidth=1.2)
    ax_alp.set_xlabel("Picard iteration")
    ax_alp.set_ylabel("alpha (under-relaxation)")
    ax_alp.set_title("Relaxation schedule")
    ax_alp.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "convergence.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot     → {png_path}")


if __name__ == "__main__":
    main()
