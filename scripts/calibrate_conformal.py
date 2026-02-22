"""Calibrate split conformal prediction intervals.

Usage
-----
    python -m scripts.calibrate_conformal --config configs/conformal.yaml

Produces
--------
- ``outputs/conformal/conformal.json``  — q_hat, alpha, timestamp
- ``outputs/conformal/summary.json``    — empirical coverage + diagnostics
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import UTC, datetime

import numpy as np
import torch
import yaml

from tokamak_transport_lab.evaluation.metrics import coverage
from tokamak_transport_lab.surrogate.conformal import SplitConformal
from tokamak_transport_lab.surrogate.ensemble import DeepEnsemble
from tokamak_transport_lab.surrogate.mlp import TransportMLP

# ── helpers ──────────────────────────────────────────────────────


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from an NPZ file produced by generate_dataset."""
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


def _load_norm_stats(path: str) -> dict[str, np.ndarray]:
    """Load normalisation stats saved during training."""
    data = np.load(path)
    return {
        "x_mean": data["x_mean"],
        "x_std": data["x_std"],
        "y_mean": float(data["y_mean"]),
        "y_std": float(data["y_std"]),
    }


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate split conformal intervals")
    parser.add_argument("--config", type=str, default="configs/conformal.yaml")
    # CLI overrides
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    conf_cfg = cfg["conformal"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    alpha = args.alpha or conf_cfg["alpha"]
    out_dir = pathlib.Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load calibration data ────────────────────────────────────
    cal_path = data_cfg["calibration"]
    print(f"Loading calibration data from {cal_path} ...")
    X_cal, y_cal = _load_npz(cal_path)
    print(f"  samples: {len(y_cal)}")

    # ── Load normalisation stats ─────────────────────────────────
    norm = _load_norm_stats(model_cfg["norm_stats"])
    X_cal_n = (X_cal - norm["x_mean"]) / norm["x_std"]

    # ── Load ensemble ────────────────────────────────────────────
    ens_dir = model_cfg["ensemble_dir"]
    n_members = model_cfg["n_members"]
    n_features = model_cfg["n_features"]
    hidden = tuple(model_cfg["hidden"])

    print(f"Loading ensemble ({n_members} members) from {ens_dir} ...")
    ensemble = DeepEnsemble.load(
        ens_dir,
        model_factory=TransportMLP,
        n_features=n_features,
        hidden=hidden,
    )

    # ── Predict on calibration set ───────────────────────────────
    with torch.no_grad():
        result = ensemble.predict(torch.from_numpy(X_cal_n))
    yhat_cal = result["mean"].squeeze(-1).numpy() * norm["y_std"] + norm["y_mean"]

    print(f"  prediction range: [{yhat_cal.min():.4f}, {yhat_cal.max():.4f}]")
    print(f"  target range:     [{y_cal.min():.4f}, {y_cal.max():.4f}]")

    # ── Fit conformal ────────────────────────────────────────────
    sc = SplitConformal()
    sc.fit(y_cal, yhat_cal, alpha=alpha)
    print(f"\nConformal calibration  alpha={alpha}  q_hat={sc.q_hat:.6f}")

    # ── Save conformal.json (augmented with timestamp) ───────────
    conformal_path = out_dir / "conformal.json"
    conformal_data = {
        "q_hat": sc.q_hat,
        "alpha": sc.alpha,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    with open(conformal_path, "w") as f:
        json.dump(conformal_data, f, indent=2)
    print(f"Saved → {conformal_path}")

    # ── Empirical coverage check ─────────────────────────────────
    lo, hi = sc.predict_interval(yhat_cal)
    cov = coverage(y_cal, lo, hi)
    print(f"Empirical coverage on calibration set: {cov:.4f}  (target ≥ {1 - alpha:.2f})")

    summary = {
        "alpha": alpha,
        "q_hat": sc.q_hat,
        "empirical_coverage": cov,
        "n_calibration": len(y_cal),
        "ensemble_dir": str(ens_dir),
        "calibration_data": str(cal_path),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
