"""Solver verification: run CN solver to steady state and compare with reference.

Usage
-----
    python -m scripts.run_solver_verification --config configs/solver.yaml

Produces
--------
- ``outputs/solver_verification.npz`` — arrays: rho, T_num, T_ref, chi, source
- ``outputs/solver_verification.png`` — matplotlib figure: numerical vs reference
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tokamak_transport_lab.physics.sources import gaussian_source
from tokamak_transport_lab.solver.analytic import steady_state_reference
from tokamak_transport_lab.solver.crank_nicolson import solve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 1-D CN solver and verify against semi-analytic reference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/solver.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_rho = cfg["grid"]["n_rho"]
    chi0 = cfg["transport"]["chi0"]
    s0 = cfg["source"]["S0"]
    rho_dep = cfg["source"]["rho_dep"]
    sigma = cfg["source"]["sigma"]
    t_ped = cfg["bc"]["Te_ped"]
    dt = cfg["solver"]["dt"]
    n_steps = cfg["solver"]["n_steps"]
    theta = cfg["solver"]["theta"]

    # ── Seed for reproducibility ─────────────────────────────────
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    # ── Run numerical solver ─────────────────────────────────────
    result = solve(
        n_rho=n_rho,
        chi0=chi0,
        s0=s0,
        rho_dep=rho_dep,
        sigma=sigma,
        t_ped=t_ped,
        dt=dt,
        n_steps=n_steps,
        theta=theta,
    )

    if not result["converged"]:
        raise SystemExit(
            f"Transient did not reach steady state: PDE residual={result['steady_residual']:.3e}"
        )

    rho = result["rho"]
    T_num = result["Te"]
    residual_history = result["residual_history"]

    # ── Compute semi-analytic reference ──────────────────────────
    T_ref = steady_state_reference(
        rho, chi0=chi0, s0=s0, rho_dep=rho_dep, sigma=sigma, t_ped=t_ped
    )

    # ── Compute metrics ──────────────────────────────────────────
    l2_err = float(np.linalg.norm(T_num - T_ref) / np.linalg.norm(T_ref - t_ped))
    linf_err = float(np.max(np.abs(T_num - T_ref)) / np.max(np.abs(T_ref - t_ped)))
    source = gaussian_source(rho, s0=s0, rho_dep=rho_dep, sigma=sigma)

    print(f"Grid points      : {n_rho}")
    print(f"Time steps       : {n_steps}")
    print(f"Final residual   : {residual_history[-1]:.3e}")
    print(f"L² rise-relative error: {l2_err:.3e}")
    print(f"L∞ rise-relative error: {linf_err:.3e}")

    # ── Save NPZ ─────────────────────────────────────────────────
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "solver_verification.npz"
    np.savez(
        npz_path,
        rho=rho,
        T_num=T_num,
        T_ref=T_ref,
        chi=result["chi"],
        source=source,
        residual_history=residual_history,
    )
    print(f"Saved NPZ  → {npz_path}")

    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Panel 1: Temperature profiles
    ax = axes[0]
    ax.plot(rho, T_num, "b-", linewidth=2, label="Numerical (CN)")
    ax.plot(rho, T_ref, "r--", linewidth=2, label="Semi-analytic ref")
    ax.set_xlabel(r"$\rho$ (normalised radius)")
    ax.set_ylabel(r"$T_e$ [eV]")
    ax.set_title("Temperature profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Point-wise error
    ax = axes[1]
    ax.semilogy(rho, np.abs(T_num - T_ref) + 1e-15, "k-", linewidth=1.5)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$|T_{\mathrm{num}} - T_{\mathrm{ref}}|$ [eV]")
    ax.set_title(f"Point-wise error  (L² rel = {l2_err:.2e})")
    ax.grid(True, alpha=0.3)

    # Panel 3: Residual convergence
    ax = axes[2]
    ax.semilogy(residual_history, "g-", linewidth=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Relative residual")
    ax.set_title("Convergence history")
    ax.grid(True, alpha=0.3)

    png_path = out_dir / "solver_verification.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {png_path}")


if __name__ == "__main__":
    main()
