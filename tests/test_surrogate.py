"""Tests for the MLP surrogate module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tokamak_transport_lab.surrogate.mlp import TransportMLP


class TestTransportMLP:
    """Basic MLP contract checks."""

    @pytest.fixture()
    def model(self) -> TransportMLP:
        return TransportMLP(n_features=6, hidden=(32, 16))

    def test_forward_shape(self, model: TransportMLP) -> None:
        """Output shape must be (batch, 1)."""
        x = torch.randn(8, 6)
        y = model(x)
        assert y.shape == (8, 1)

    def test_output_positive(self, model: TransportMLP) -> None:
        """Softplus guarantees Q ≥ 0."""
        x = torch.randn(100, 6)
        y = model(x)
        assert (y >= 0).all()

    def test_output_finite(self, model: TransportMLP) -> None:
        """No NaN or Inf in output."""
        x = torch.randn(64, 6)
        y = model(x)
        assert torch.isfinite(y).all()

    def test_save_load_roundtrip(self, model: TransportMLP) -> None:
        """Model can be saved and loaded, reproducing the same output."""
        x = torch.randn(4, 6)
        model.eval()
        with torch.no_grad():
            y_before = model(x).clone()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = TransportMLP.load(path, n_features=6, hidden=(32, 16))

        with torch.no_grad():
            y_after = loaded(x)
        np.testing.assert_allclose(
            y_before.numpy(), y_after.numpy(), atol=1e-6
        )

    def test_single_sample(self, model: TransportMLP) -> None:
        """Works with batch size 1."""
        x = torch.randn(1, 6)
        y = model(x)
        assert y.shape == (1, 1)
