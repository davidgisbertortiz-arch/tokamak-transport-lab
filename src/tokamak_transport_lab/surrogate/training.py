"""CPU training with train-only transforms and validation checkpoint selection."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from tokamak_transport_lab.data.generate import FEATURE_NAMES
from tokamak_transport_lab.evaluation.metrics import mae, max_error, r2_score
from tokamak_transport_lab.surrogate.bundle import TransportBundle
from tokamak_transport_lab.surrogate.mlp import TransportMLP


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["target"]) != "chi" or data["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Regenerate data using the version-2 dataset schema")
        return data["X"].copy(), data["y"].copy(), data["bounds"].copy()


def regression_metrics(y, pred):
    return {
        "r2": r2_score(y, pred),
        "mae": mae(y, pred),
        "max_error": max_error(y, pred),
        "median_relative_error": float(np.median(np.abs(pred - y) / y)),
        "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(y)) ** 2))),
    }


def train_bundle(
    train_path,
    validation_path,
    out_dir,
    *,
    seeds=(42,),
    hidden=(64, 64, 64),
    epochs=200,
    batch_size=512,
    lr=0.002,
):
    torch.set_num_threads(1)
    x, y, bounds = load_data(train_path)
    xv, yv, _ = load_data(validation_path)
    if Path(train_path).resolve() == Path(validation_path).resolve():
        raise ValueError("Training and validation must be separate")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-12)
    ym, ys = np.log(y).mean(), np.log(y).std()
    xt = torch.tensor((x - xm) / xs, dtype=torch.float32)
    yt = torch.tensor((np.log(y) - ym) / ys, dtype=torch.float32)[:, None]
    vt = torch.tensor((xv - xm) / xs, dtype=torch.float32)
    vy = torch.tensor((np.log(yv) - ym) / ys, dtype=torch.float32)[:, None]
    active_val = torch.tensor(xv[:, 0] > xv[:, 2])
    models, histories = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = TransportMLP(n_features=5, hidden=hidden, output_activation="linear")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best, state, best_epoch = float("inf"), None, 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            for idx in torch.randperm(len(xt)).split(batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xt[idx]), yt[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val = float(torch.nn.functional.mse_loss(model(vt)[active_val], vy[active_val]))
            history.append(val)
            if val < best:
                best, state, best_epoch = val, deepcopy(model.state_dict()), epoch
            if epoch == 1 or epoch % 50 == 0:
                print(
                    f"seed={seed} epoch={epoch}/{epochs} validation_log_mse={val:.6g}", flush=True
                )
        model.load_state_dict(state)
        models.append(model)
        histories.append({"seed": seed, "best_epoch": best_epoch, "validation_loss": history})
    bundle = TransportBundle(
        models, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, bounds=bounds, hidden=hidden, seeds=seeds
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle.save(out / "model.pt")
    metrics = regression_metrics(yv, bundle.predict_chi(xv))
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata = {
        "schema_version": 2,
        "features": FEATURE_NAMES,
        "target": "chi",
        "transform": "standardized log chi",
        "seeds": list(seeds),
        "epochs": epochs,
        "hidden": list(hidden),
        "n_train": len(x),
        "n_validation": len(xv),
        "train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
        "artifact_sha256": bundle.fingerprint,
        "torch_version": str(torch.__version__),
        "histories": histories,
    }
    (out / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {out / 'model.pt'}; validation metrics: {metrics}", flush=True)
    return bundle
