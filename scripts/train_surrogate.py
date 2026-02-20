"""Train a single-MLP transport surrogate.

Usage
-----
    python -m scripts.train_surrogate \\
        --train data/transport_train.npz \\
        --test  data/transport_test.npz \\
        --output-dir outputs/surrogate

Produces
--------
- ``model.pt``         — trained weights
- ``metrics.json``     — R², MAE, max-error on test set
- ``calibration.png``  — predicted-vs-true scatter plot
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tokamak_transport_lab.evaluation.metrics import mae, max_error, r2_score
from tokamak_transport_lab.surrogate.mlp import TransportMLP
from tokamak_transport_lab.utils.seeding import seed_everything


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from an NPZ file produced by generate_dataset."""
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transport MLP surrogate")
    parser.add_argument("--train", type=str, default="data/transport_train.npz")
    parser.add_argument("--test", type=str, default="data/transport_test.npz")
    parser.add_argument("--output-dir", type=str, default="outputs/surrogate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    X_train, y_train = _load_npz(args.train)
    X_test, y_test = _load_npz(args.test)

    # Compute normalisation stats on train set
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_train_n = (X_train - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(y_train_n).unsqueeze(-1),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # ── Model ────────────────────────────────────────────────────
    n_features = X_train.shape[1]
    model = TransportMLP(n_features=n_features, hidden=(128, 128, 64))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────
    model.train()
    for epoch in range(1, args.epochs + 1):
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

        if epoch % 20 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}")

    # ── Evaluate ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        # Predictions on normalised inputs, then de-normalise
        y_pred_n = model(torch.from_numpy(X_test_n)).squeeze(-1).numpy()

    # The model predicts normalised y through a Softplus, so we need to
    # un-normalise.  Because we trained on (y - y_mean)/y_std, the raw
    # model output is in normalised space.  De-normalise:
    y_pred = y_pred_n * y_std + y_mean
    y_true = y_test

    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "epochs": args.epochs,
    }
    print(f"\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    # ── Save model ───────────────────────────────────────────────
    model_path = out_dir / "model.pt"
    model.save(model_path)
    print(f"\nModel   → {model_path}")

    # Save normalisation stats alongside model
    np.savez(
        out_dir / "norm_stats.npz",
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.float32(y_mean),
        y_std=np.float32(y_std),
    )

    # ── Save metrics ─────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics → {metrics_path}")

    # ── Calibration plot ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="ideal")
    ax.set_xlabel("True Q_norm")
    ax.set_ylabel("Predicted Q_norm")
    ax.set_title(f"Calibration  (R² = {metrics['r2']:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = out_dir / "calibration.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot    → {png_path}")


if __name__ == "__main__":
    main()
