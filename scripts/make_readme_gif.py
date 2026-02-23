"""Generate the README demo GIF headlessly (no Streamlit needed).

Usage::

    python -m scripts.make_readme_gif
    python -m scripts.make_readme_gif --param S0_e --start 0.5 --end 8 --frames 12
    python -m scripts.make_readme_gif --out assets/demo.gif

Requires::

    pip install -e ".[gif]"
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README demo GIF")
    parser.add_argument(
        "--param",
        type=str,
        default="S0_e",
        choices=["S0_e", "te_ped", "ti_ped", "density", "tau_eq"],
        help="Parameter to sweep (default: S0_e)",
    )
    parser.add_argument("--start", type=float, default=0.5, help="Sweep start value")
    parser.add_argument("--end", type=float, default=6.0, help="Sweep end value")
    parser.add_argument("--frames", type=int, default=10, help="Number of GIF frames")
    parser.add_argument(
        "--out",
        type=str,
        default="assets/demo.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.6,
        help="Seconds per frame",
    )
    args = parser.parse_args()

    from tokamak_transport_lab.app.gif_export import generate_gif

    out = generate_gif(
        param=args.param,
        start=args.start,
        end=args.end,
        n_frames=args.frames,
        out_path=args.out,
        duration_s=args.duration,
    )
    print(f"GIF saved → {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
