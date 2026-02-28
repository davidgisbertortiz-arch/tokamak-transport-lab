"""Generate the README demo GIF headlessly (no Streamlit needed).

Usage::

    python -m scripts.make_readme_gif
    python -m scripts.make_readme_gif --config configs/scenarios/mid_power.yaml --n_frames 10
    python -m scripts.make_readme_gif --param S0_e --start 0.5 --end 8 --n_frames 12
    python -m scripts.make_readme_gif --out assets/demo.gif

Requires::

    pip install -e ".[gif]"
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate README demo GIF (P_heat sweep)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML scenario file (overrides --te_ped, --ti_ped etc.)",
    )
    parser.add_argument(
        "--param",
        type=str,
        default="P_heat",
        choices=["P_heat", "S0_e", "te_ped", "ti_ped", "density", "tau_eq"],
        help="Parameter to sweep (default: P_heat)",
    )
    parser.add_argument("--start", type=float, default=1.0, help="Sweep start value")
    parser.add_argument("--end", type=float, default=15.0, help="Sweep end value")
    parser.add_argument("--n_frames", type=int, default=12, help="Number of GIF frames")
    parser.add_argument(
        "--out",
        type=str,
        default="assets/demo.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.45,
        help="Seconds per frame",
    )
    args = parser.parse_args()

    # ── load YAML config if provided ─────────────────────────────
    base_kwargs: dict[str, float] = {}
    if args.config is not None:
        import yaml

        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)

        # Only inherit grid + picard settings from the scenario config.
        # We do NOT inherit te_ped / transport params — those stay at
        # GIF-friendly defaults (lower pedestal, softer stiffness) so
        # the sweep produces visually distinct profiles.
        grid = cfg.get("grid", {})
        n_rho = grid.get("n_rho", 50)
        # Cap at 60 to keep each frame fast
        base_kwargs["n_rho"] = min(int(n_rho), 60)

        picard = cfg.get("picard", {})
        base_kwargs["max_iters"] = picard.get("max_iters", 80)
        # Use looser tol for GIF — convergence display is still real
        base_kwargs["tol"] = max(float(picard.get("tol", 5e-4)), 5e-4)

    from tokamak_transport_lab.app.gif_export import generate_gif

    out = generate_gif(
        param=args.param,
        start=args.start,
        end=args.end,
        n_frames=args.n_frames,
        out_path=args.out,
        duration_s=args.duration,
        base_kwargs=base_kwargs if base_kwargs else None,
    )
    size_kb = out.stat().st_size / 1024
    print(f"GIF saved → {out}  ({size_kb:.0f} KB)")
    if size_kb > 5120:
        print("⚠  GIF exceeds 5 MB target — consider reducing --n_frames or --duration.")


if __name__ == "__main__":
    main()
