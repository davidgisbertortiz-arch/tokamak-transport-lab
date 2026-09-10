"""Generate disjoint train, validation, calibration and test datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tokamak_transport_lab.data.generate import FEATURE_NAMES, generate_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--size_train", type=int)
    parser.add_argument("--size_test", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sampling = cfg["sampling"]
    bounds = [sampling["bounds"][name] for name in FEATURE_NAMES]
    sizes = dict(cfg["sizes"])
    if args.size_train is not None:
        sizes["train"] = args.size_train
    if args.size_test is not None:
        sizes["test"] = args.size_test
    out = Path(args.output_dir or cfg["output_dir"])
    for offset, split in enumerate(("train", "validation", "calibration", "test")):
        data = generate_dataset(
            sizes[split],
            bounds=bounds,
            seed=sampling["seed"] + offset,
            sampling="lhs" if split == "train" else "iid",
        )
        save_dataset(data, out / f"transport_{split}.npz")
        print(
            f"{split}: {data['X'].shape}, chi range [{data['y'].min():.4g}, {data['y'].max():.4g}]"
        )


if __name__ == "__main__":
    main()
