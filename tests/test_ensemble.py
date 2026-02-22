"""Tests for the DeepEnsemble wrapper."""

from __future__ import annotations

import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from tokamak_transport_lab.surrogate.ensemble import DeepEnsemble
from tokamak_transport_lab.surrogate.mlp import TransportMLP


def _make_ensemble(n: int = 3) -> DeepEnsemble:
    """Create a small ensemble of randomly-initialised MLPs."""
    return DeepEnsemble(
        TransportMLP,
        n_members=n,
        n_features=6,
        hidden=(16, 8),
    )


class TestDeepEnsemble:
    """DeepEnsemble contract checks."""

    def test_predict_shapes(self) -> None:
        ens = _make_ensemble(3)
        x = torch.randn(10, 6)
        result = ens.predict(x)
        assert result["mean"].shape == (10, 1)
        assert result["epistemic_std"].shape == (10, 1)
        assert result["member_means"].shape == (3, 10, 1)

    def test_epistemic_std_positive(self) -> None:
        """With different random inits, std should be > 0 almost everywhere."""
        ens = _make_ensemble(5)
        x = torch.randn(50, 6)
        result = ens.predict(x)
        assert (result["epistemic_std"] > 0).float().mean() > 0.5

    def test_predict_members(self) -> None:
        ens = _make_ensemble(4)
        x = torch.randn(8, 6)
        preds = ens.predict_members(x)
        assert preds.shape == (4, 8, 1)
        assert torch.isfinite(preds).all()

    def test_len(self) -> None:
        assert len(_make_ensemble(3)) == 3

    def test_save_load_roundtrip(self) -> None:
        ens = _make_ensemble(2)
        x = torch.randn(4, 6)
        before = ens.predict(x)["mean"]

        with tempfile.TemporaryDirectory() as tmp:
            ens.save(tmp)
            loaded = DeepEnsemble.load(
                tmp,
                model_factory=TransportMLP,
                n_features=6,
                hidden=(16, 8),
            )

        after = loaded.predict(x)["mean"]
        assert torch.allclose(before, after, atol=1e-6)

    def test_single_member(self) -> None:
        """Ensemble with 1 member should still work; std = 0."""
        ens = _make_ensemble(1)
        x = torch.randn(5, 6)
        result = ens.predict(x)
        assert result["mean"].shape == (5, 1)
        assert (result["epistemic_std"] == 0).all()

    def test_fit_reduces_loss(self) -> None:
        """fit() should reduce training loss over a toy regression task."""
        ens = DeepEnsemble(
            TransportMLP,
            n_members=2,
            seeds=[0, 1],
            n_features=6,
            hidden=(16, 8),
        )
        torch.manual_seed(42)
        x = torch.randn(64, 6)
        y = x[:, 0:1] * 2 + x[:, 1:2]
        dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        histories = ens.fit(dl, epochs=30, lr=1e-2)
        for key, hist in histories.items():
            losses = hist["train_loss"]
            assert losses[-1] < losses[0], f"{key}: final >= initial"
