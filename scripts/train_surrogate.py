"""Train a versioned diffusivity surrogate; reserve test data for evaluation."""

from __future__ import annotations

import argparse

from tokamak_transport_lab.surrogate.training import train_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/transport_train.npz")
    parser.add_argument("--validation", default="data/transport_validation.npz")
    parser.add_argument("--output-dir", default="outputs/surrogate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_bundle(
        args.train,
        args.validation,
        args.output_dir,
        seeds=[args.seed],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
