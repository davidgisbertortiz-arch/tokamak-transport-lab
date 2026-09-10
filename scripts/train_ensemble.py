"""Train independent members and save one complete inference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tokamak_transport_lab.surrogate.training import train_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ensemble.yaml")
    parser.add_argument("--n-members", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    n = args.n_members if args.n_members is not None else cfg["ensemble"]["n_members"]
    if n < 2:
        raise ValueError("An uncertainty ensemble requires at least two members")
    seeds = [cfg["ensemble"]["base_seed"] + i for i in range(n)]
    training = dict(cfg["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    train_bundle(
        cfg["data"]["train"],
        cfg["data"]["validation"],
        args.output_dir or cfg["output_dir"],
        seeds=seeds,
        hidden=tuple(cfg["model"]["hidden"]),
        **training,
    )


if __name__ == "__main__":
    main()
