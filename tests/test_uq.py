"""UQ tests: DeepEnsemble + SplitConformal (fast, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tokamak_transport_lab.surrogate.conformal import SplitConformal
from tokamak_transport_lab.surrogate.ensemble import DeepEnsemble

# ── DeepEnsemble with tiny Linear members ────────────────────────


def _make_linear_ensemble(n_members: int = 3, seed: int = 0) -> DeepEnsemble:
    """Build a DeepEnsemble from tiny 2→1 Linear modules with *different* weights."""
    return DeepEnsemble(
        model_factory=lambda: nn.Linear(2, 1),
        n_members=n_members,
        seeds=[seed + i for i in range(n_members)],
    )


class TestDeepEnsembleUQ:
    """Epistemic std must be positive when members disagree."""

    def test_epistemic_std_positive_toy(self) -> None:
        """2-3 tiny Linear members with different inits → std > 0."""
        ens = _make_linear_ensemble(n_members=3, seed=7)
        torch.manual_seed(99)
        x = torch.randn(8, 2)
        result = ens.predict(x)
        std = result["epistemic_std"]
        assert std.shape == (8, 1)
        assert (std > 0).all(), f"Expected all std > 0, got {std.squeeze().tolist()}"

    def test_mean_is_finite(self) -> None:
        ens = _make_linear_ensemble(n_members=2, seed=0)
        x = torch.tensor([[1.0, 2.0], [-0.5, 0.3]])
        result = ens.predict(x)
        assert torch.isfinite(result["mean"]).all()
        assert torch.isfinite(result["epistemic_std"]).all()

    def test_fit_reduces_loss(self) -> None:
        """fit() should reduce training loss over a toy regression task."""
        ens = DeepEnsemble(
            model_factory=lambda: nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1),
            ),
            n_members=2,
            seeds=[0, 1],
        )
        torch.manual_seed(42)
        x = torch.randn(64, 2)
        y = (x[:, 0:1] * 2 + x[:, 1:2]).float()
        dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        histories = ens.fit(dl, epochs=30, lr=1e-2)
        for key, hist in histories.items():
            losses = hist["train_loss"]
            assert losses[-1] < losses[0], f"{key}: final loss >= initial"


# ── SplitConformal coverage on synthetic data ────────────────────


class TestSplitConformalCoverage:
    """Coverage on a held-out test set should be ≈ 1 − alpha."""

    @pytest.mark.parametrize("alpha", [0.1, 0.2])
    def test_coverage_near_target(self, alpha: float) -> None:
        """y = sin(x) + noise; yhat = sin(x).  Calibrate, then check test coverage."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(-3, 3, size=n)
        noise_std = 0.3
        y = np.sin(x) + rng.normal(0, noise_std, size=n)
        yhat = np.sin(x)  # perfect model ignoring noise

        # Split: first half calibration, second half test
        n_cal = n // 2
        sc = SplitConformal()
        sc.fit(y[:n_cal], yhat[:n_cal], alpha=alpha)

        lo, hi = sc.predict_interval(yhat[n_cal:])
        cov = float(np.mean((y[n_cal:] >= lo) & (y[n_cal:] <= hi)))

        target = 1.0 - alpha
        assert abs(cov - target) < 0.05, (
            f"alpha={alpha}: coverage={cov:.3f}, expected ≈ {target:.2f}"
        )

    def test_torch_tensor_input(self) -> None:
        """SplitConformal should accept torch tensors transparently."""
        rng = np.random.default_rng(0)
        n = 500
        y_np = rng.normal(0, 1, size=n)
        yhat_np = y_np + rng.normal(0, 0.1, size=n)

        y_t = torch.from_numpy(y_np)
        yhat_t = torch.from_numpy(yhat_np)

        sc = SplitConformal()
        sc.fit(y_t, yhat_t, alpha=0.1)
        assert sc.q_hat is not None and sc.q_hat > 0

        lo, _hi = sc.predict_interval(yhat_t)
        assert isinstance(lo, np.ndarray)
        assert lo.shape == (n,)
