"""Run a stationary single- or two-channel experiment from YAML."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tokamak_transport_lab.integration.config import (
    build_transport,
    geometry_function,
    run_config,
    transport_params,
    validate_config,
)
from tokamak_transport_lab.integration.picard import run_picard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/picard.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_config(cfg)
    coupled = "Ti_ped" in cfg.get("bc", {}) or "transport_e" in cfg
    out = Path(args.output_dir or cfg.get("output_dir", "outputs/picard")) / (
        args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    if coupled:
        result = run_config(cfg)
        arrays = {
            "rho": result.rho,
            "Te": result.te_final,
            "Ti": result.ti_final,
            "chi_e": result.chi_e_profile,
            "chi_i": result.chi_i_profile,
        }
    else:
        model, bundle, mode = build_transport(cfg)
        src = cfg.get("source", {})
        result = run_picard(
            n_rho=cfg.get("grid", {}).get("n_rho", 100),
            t_ped=cfg.get("bc", {}).get("Te_ped", 500),
            s0=src.get("S0", 1),
            rho_dep=src.get("rho_dep", 0.3),
            sigma=src.get("sigma", 0.1),
            v_prime_fn=geometry_function(cfg),
            transport_model=model,
            transport_params=transport_params(cfg),
            **cfg.get("picard", {}),
        )
        result.metadata.update(
            transport_model=mode,
            artifact_sha256=bundle.fingerprint if bundle else None,
            out_of_domain_fraction=getattr(model, "out_of_domain_fraction", 0.0),
        )
        arrays = {"rho": result.rho, "Te": result.te_final, "chi": result.chi_profile}
    arrays.update(residual_history=result.residual_history, alpha_history=result.alpha_history)
    np.savez_compressed(out / "result.npz", **arrays)
    (out / "metrics.json").write_text(
        json.dumps(result.metadata, indent=2, allow_nan=False) + "\n"
    )
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(result.rho, result.te_final, label="Te")
    if coupled:
        axes[0].plot(result.rho, result.ti_final, label="Ti")
    axes[0].set(xlabel="Normalized radius", ylabel="Temperature [eV]")
    axes[0].legend()
    axes[1].semilogy(result.residual_history)
    axes[1].set(xlabel="Nonlinear iteration", ylabel="Source-scaled PDE residual")
    for key, value in arrays.items():
        if key.startswith("chi"):
            axes[2].plot(result.rho, value, label=key)
    axes[2].set(xlabel="Normalized radius", ylabel="Normalized diffusivity")
    axes[2].legend()
    fig.savefig(out / "profiles.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result.metadata, indent=2))
    print(f"Outputs: {out}")
    if not result.metadata["converged"]:
        raise SystemExit("Stationary solve did not converge; outputs contain the last iterate.")


if __name__ == "__main__":
    main()
