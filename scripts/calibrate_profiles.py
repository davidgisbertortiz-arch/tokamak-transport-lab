"""Calibrate and test simultaneous Te/Ti bands on independent random scenarios."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from tokamak_transport_lab.integration.config import run_config
from tokamak_transport_lab.surrogate.bundle import TransportBundle
from tokamak_transport_lab.surrogate.conformal import SplitConformal
from tokamak_transport_lab.surrogate.profile_uq import FAMILY, sample_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/ensemble/model.pt")
    parser.add_argument("--n-calibration", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs/profile_uq")
    args = parser.parse_args()
    bundle = TransportBundle.load(args.model)
    all_data = {}
    for split, count, seed in [
        ("calibration", args.n_calibration, 701),
        ("test", args.n_test, 702),
    ]:
        errors, truth, predictions, parameters = [], [], [], []
        for i, cfg in enumerate(sample_configs(count, seed)):
            reference = run_config(cfg)
            model_cfg = deepcopy(cfg)
            model_cfg["model"] = {"type": "ensemble", "path": args.model}
            pred = run_config(model_cfg, bundle=bundle)
            if not (reference.metadata["converged"] and pred.metadata["converged"]):
                raise RuntimeError(
                    f"{split} case {i} failed; no cases are discarded: {reference.metadata}, {pred.metadata}"
                )
            target = np.array([reference.te_final, reference.ti_final])
            estimate = np.array([pred.te_final, pred.ti_final])
            errors.append(float(np.max(abs(target - estimate))))
            truth.append(target)
            predictions.append(estimate)
            parameters.append([cfg["source"]["S0_e"], cfg["bc"]["Te_ped"]])
            if (i + 1) % 8 == 0:
                print(f"{split}: {i + 1}/{count} scenarios completed", flush=True)
        all_data[split] = {
            "scores": np.array(errors),
            "truth": np.array(truth),
            "prediction": np.array(predictions),
            "parameters": np.array(parameters),
            "seed": seed,
        }
    cal = all_data["calibration"]["scores"]
    sc = SplitConformal().fit(cal, np.zeros_like(cal), alpha=args.alpha)
    scores = all_data["test"]["scores"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "target": "simultaneous_Te_Ti_max_abs_eV",
        "alpha": args.alpha,
        "q_hat_eV": sc.q_hat,
        "test_simultaneous_coverage": float(np.mean(scores <= sc.q_hat)),
        "n_calibration": len(cal),
        "n_test": len(scores),
        "family": FAMILY,
        "artifact_sha256": bundle.fingerprint,
        "seeds": {"calibration": 701, "test": 702},
        "test_max_profile_error_eV": float(scores.max()),
        "test_median_profile_error_eV": float(np.median(scores)),
        "scope": "Marginal over independent uniform draws of source and pedestal in the specified synthetic family; fixed grid; no device or conditional coverage claim",
    }
    (out / "calibration.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for split, data in all_data.items():
        np.savez_compressed(out / f"{split}_profiles.npz", **data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
