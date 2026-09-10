"""Reproduce the trained models, calibration experiments and validation outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-training", action="store_true", help="Reuse existing data and model bundles"
    )
    parser.add_argument(
        "--gif", action="store_true", help="Also render outputs/reproduction/demo.gif"
    )
    args = parser.parse_args()
    commands = []
    if not args.skip_training:
        commands += [
            ["scripts.generate_dataset"],
            ["scripts.train_surrogate"],
            ["scripts.train_ensemble"],
        ]
    commands += [
        ["scripts.run_solver_verification"],
        ["scripts.run_scenarios"],
        ["scripts.calibrate_conformal"],
        ["scripts.calibrate_profiles"],
        ["scripts.validate_project"],
    ]
    if args.gif:
        commands.append(["scripts.make_readme_gif", "--out", "outputs/reproduction/demo.gif"])
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    for command in commands:
        print("Running: " + " ".join(command), flush=True)
        subprocess.run([sys.executable, "-m", *command], check=True, env=env)


if __name__ == "__main__":
    main()
