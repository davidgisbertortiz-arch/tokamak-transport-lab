"""Calibrate log-diffusivity intervals and evaluate on an independent IID test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from tokamak_transport_lab.surrogate.bundle import TransportBundle
from tokamak_transport_lab.surrogate.conformal import SplitConformal
from tokamak_transport_lab.surrogate.training import load_data, regression_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conformal.yaml")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cal_path, test_path = map(Path, (cfg["data"]["calibration"], cfg["data"]["test"]))
    if (
        cal_path.resolve() == test_path.resolve()
        or cal_path.read_bytes() == test_path.read_bytes()
    ):
        raise ValueError("Calibration and test sets must be independent files")
    x, y, bounds = load_data(cal_path)
    xt, yt, test_bounds = load_data(test_path)
    for path in (cal_path, test_path):
        with np.load(path, allow_pickle=False) as d:
            if str(d["sampling"]) != "iid":
                raise ValueError("Conformal evaluation requires IID calibration/test sampling")
    if not np.array_equal(bounds, test_bounds):
        raise ValueError("Calibration/test distributions differ")
    bundle = TransportBundle.load(cfg["model"]["path"])
    alpha = cfg["conformal"]["alpha"] if args.alpha is None else args.alpha
    predictor = SplitConformal().fit(np.log(y), np.log(bundle.predict_chi(x)), alpha=alpha)
    pred = bundle.predict_chi(xt)
    lo, hi = predictor.interval(np.log(pred))
    lo, hi = np.exp(lo), np.exp(hi)
    # Intersect with the known support of this specified synthetic closure.
    lo = np.maximum(lo, xt[:, -1])
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    predictor.save(out / "conformal.json")
    saved = json.loads((out / "conformal.json").read_text())
    saved.update(
        target="log_chi",
        artifact_sha256=bundle.fingerprint,
        bounds=bounds.tolist(),
        calibration_sha256=hashlib.sha256(cal_path.read_bytes()).hexdigest(),
    )
    (out / "conformal.json").write_text(json.dumps(saved, indent=2) + "\n")
    summary = {
        "alpha": alpha,
        "q_hat_log_chi": predictor.q_hat,
        "n_calibration": len(y),
        "n_test": len(yt),
        "test_coverage": float(np.mean((yt >= lo) & (yt <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
        "test_metrics": regression_metrics(yt, pred),
        "assumption": "IID inputs from the declared uniform closure-parameter box",
        "temperature_coverage_claim": False,
        "artifact_sha256": bundle.fingerprint,
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        out / "test_predictions.npz", X=xt, y=yt, prediction=pred, lower=lo, upper=hi
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
