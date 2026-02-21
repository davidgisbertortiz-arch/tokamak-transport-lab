"""Train a Deep Ensemble of MLP transport surrogates.

Usage
-----
    python -m scripts.train_ensemble --config configs/ensemble.yaml

Produces
--------
- ``outputs/ensemble/member_0.pt`` … ``member_{N-1}.pt``
- ``outputs/ensemble/norm_stats.npz``  — shared normalisation statistics
- ``outputs/ensemble/metadata.json``   — seeds, config, git hash
- ``outputs/ensemble/metrics.json``    — per-member + ensemble test metrics
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from tokamak_transport_lab.evaluation.metrics import mae, max_error, r2_score
from tokamak_transport_lab.surrogate.ensemble import DeepEnsemble
from tokamak_transport_lab.surrogate.mlp import TransportMLP
from tokamak_transport_lab.utils.seeding import seed_everything

# ── helpers ──────────────────────────────────────────────────────


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from an NPZ file produced by generate_dataset."""
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


def _git_hash() -> str | None:
    """Return short git hash, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _train_single(
    model: nn.Module,
    train_dl: DataLoader,
    *,
    epochs: int,
    lr: float,
    member_id: int,
) -> list[float]:
    """Train a single MLP member; return per-epoch losses."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            optimiser.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    member {member_id}  epoch {epoch:4d}/{epochs}  loss={avg:.6f}")

    return history


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep Ensemble")
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml")
    # CLI overrides
    parser.add_argument("--n-members", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ens_cfg = cfg["ensemble"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    n_members = args.n_members or ens_cfg["n_members"]
    base_seed = ens_cfg["base_seed"]
    n_features = model_cfg["n_features"]
    hidden = tuple(model_cfg["hidden"])
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    out_dir = pathlib.Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    X_train, y_train = _load_npz(data_cfg["train"])
    X_test, y_test = _load_npz(data_cfg["test"])

    # Normalisation (shared across all members)
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) + 1e-8

    X_train_n = (X_train - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std

    xt = torch.from_numpy(X_train_n)
    yt = torch.from_numpy(y_train_n).unsqueeze(-1)

    np.savez(
        out_dir / "norm_stats.npz",
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.float32(y_mean),
        y_std=np.float32(y_std),
    )
    print(f"Norm stats → {out_dir / 'norm_stats.npz'}")

    # ── Train each member ────────────────────────────────────────
    members: list[nn.Module] = []
    seeds_used: list[int] = []
    member_metrics: list[dict] = []

    for i in range(n_members):
        seed_i = base_seed + i
        seeds_used.append(seed_i)
        print(f"\n── Member {i}/{n_members}  seed={seed_i} ──")
        seed_everything(seed_i)

        model = TransportMLP(n_features=n_features, hidden=hidden)
        dl = DataLoader(
            TensorDataset(xt, yt),
            batch_size=batch_size,
            shuffle=True,
        )
        _train_single(model, dl, epochs=epochs, lr=lr, member_id=i)

        # Save individual checkpoint
        ckpt_path = out_dir / f"member_{i}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"    saved → {ckpt_path}")

        # Evaluate individual member
        model.eval()
        with torch.no_grad():
            y_pred_n = model(torch.from_numpy(X_test_n)).squeeze(-1).numpy()
        y_pred = y_pred_n * y_std + y_mean
        member_metrics.append(
            {
                "member": i,
                "seed": seed_i,
                "r2": r2_score(y_test, y_pred),
                "mae": mae(y_test, y_pred),
                "max_error": max_error(y_test, y_pred),
            }
        )
        members.append(model)

    # ── Ensemble evaluation ──────────────────────────────────────
    ensemble = DeepEnsemble(members)
    result = ensemble.predict(torch.from_numpy(X_test_n))
    ens_mean = result["mean"].squeeze(-1).numpy() * y_std + y_mean
    ens_std = result["epistemic_std"].squeeze(-1).numpy() * y_std

    ens_metrics = {
        "r2": r2_score(y_test, ens_mean),
        "mae": mae(y_test, ens_mean),
        "max_error": max_error(y_test, ens_mean),
        "mean_epistemic_std": float(ens_std.mean()),
    }

    all_metrics = {
        "ensemble": ens_metrics,
        "members": member_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics → {metrics_path}")

    print("\nEnsemble test metrics:")
    for k, v in ens_metrics.items():
        print(f"  {k:22s}: {v}")

    # ── Metadata ─────────────────────────────────────────────────
    metadata = {
        "n_members": n_members,
        "seeds": seeds_used,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden": list(hidden),
        "n_features": n_features,
        "git_hash": _git_hash(),
        "config_file": str(args.config),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
