"""Run transient diffusion, save output, and report stationary balance."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml

from tokamak_transport_lab.solver.crank_nicolson import solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1-D diffusion solver")
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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = solve(
        n_rho=cfg["grid"]["n_rho"],
        chi0=cfg["transport"]["chi0"],
        s0=cfg["source"]["S0"],
        rho_dep=cfg["source"]["rho_dep"],
        sigma=cfg["source"]["sigma"],
        t_ped=cfg["bc"]["Te_ped"],
        dt=cfg["solver"]["dt"],
        n_steps=cfg["solver"]["n_steps"],
        theta=cfg["solver"]["theta"],
    )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "solver_verification.npz",
        rho=result["rho"],
        Te=result["Te"],
        chi=result["chi"],
        source=result["source"],
        residual_history=result["residual_history"],
        steady_residual=result["steady_residual"],
        converged=result["converged"],
        time=result["time"],
    )
    print(f"Stationary: {result['converged']}; PDE residual: {result['steady_residual']:.3e}")
    print(f"Saved results to {out_dir / 'solver_verification.npz'}")


if __name__ == "__main__":
    main()
