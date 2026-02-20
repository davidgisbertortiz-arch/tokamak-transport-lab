"""Generate synthetic transport datasets.

Usage
-----
    python -m scripts.generate_dataset --config configs/dataset.yaml

Produces
--------
- ``data/transport_train.npz`` — training set
- ``data/transport_test.npz``  — test set
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

from tokamak_transport_lab.data.generate import generate_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transport dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--size_train", type=int, default=None, help="Override train size")
    parser.add_argument("--size_test", type=int, default=None, help="Override test size")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sampling = cfg["sampling"]
    sizes = cfg["sizes"]

    size_train = args.size_train or sizes["train"]
    size_test = args.size_test or sizes["test"]
    output_dir = pathlib.Path(args.output_dir or cfg.get("output_dir", "data"))

    # Build bounds list in canonical order
    bounds_cfg = sampling["bounds"]
    bounds_order = [
        "a_over_LTe",
        "q",
        "s_hat",
        "nu_star",
        "chi_s",
        "a_over_LTe_crit",
    ]
    bounds = [tuple(bounds_cfg[k]) for k in bounds_order]

    alpha_s = sampling.get("alpha_s", 1.5)
    chi_neo = sampling.get("chi_neo", 0.01)
    seed = sampling.get("seed", 42)

    # ── Generate train set ───────────────────────────────────────
    print(f"Generating training set ({size_train} samples)...")
    train_data = generate_dataset(
        size_train,
        bounds=bounds,
        alpha_s=alpha_s,
        chi_neo=chi_neo,
        seed=seed,
    )
    train_path = output_dir / "transport_train.npz"
    save_dataset(train_data, train_path)
    print(f"  Saved → {train_path}")

    # ── Generate test set (different seed) ───────────────────────
    print(f"Generating test set ({size_test} samples)...")
    test_data = generate_dataset(
        size_test,
        bounds=bounds,
        alpha_s=alpha_s,
        chi_neo=chi_neo,
        seed=seed + 1,
    )
    test_path = output_dir / "transport_test.npz"
    save_dataset(test_data, test_path)
    print(f"  Saved → {test_path}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\nDone. Feature names: {list(train_data['feature_names'])}")
    print(f"  Train X shape: {train_data['X'].shape}, y range: "
          f"[{train_data['y'].min():.2f}, {train_data['y'].max():.2f}]")
    print(f"  Test  X shape: {test_data['X'].shape},  y range: "
          f"[{test_data['y'].min():.2f}, {test_data['y'].max():.2f}]")


if __name__ == "__main__":
    main()
