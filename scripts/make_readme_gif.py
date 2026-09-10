"""Generate a GIF from actual converged runs of the supplied configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tokamak_transport_lab.app.gif_export import generate_gif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--param", default="S0_e", choices=["S0_e", "te_ped", "ti_ped", "density", "tau_eq"]
    )
    parser.add_argument("--start", type=float, default=300.0)
    parser.add_argument("--end", type=float, default=1000.0)
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--duration", type=float, default=0.6)
    parser.add_argument("--out", default="assets/demo.gif")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else None
    out = generate_gif(
        config=cfg,
        param=args.param,
        start=args.start,
        end=args.end,
        n_frames=args.n_frames,
        duration_s=args.duration,
        out_path=args.out,
    )
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KiB) and its JSON provenance")


if __name__ == "__main__":
    main()
