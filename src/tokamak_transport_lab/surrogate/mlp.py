"""Single MLP surrogate for normalised heat-flux prediction.

Architecture
------------
input(n_features) → 128 → 128 → 64 → 1  (ReLU hidden, Softplus output)

The softplus output guarantees Qe/Q_gB ≥ 0.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class TransportMLP(nn.Module):
    """Feedforward MLP mapping local plasma parameters → Q_norm.

    Parameters
    ----------
    n_features : int
        Number of input features (default 6).
    hidden : tuple[int, ...]
        Hidden-layer widths.
    """

    def __init__(
        self,
        n_features: int = 6,
        hidden: tuple[int, ...] = (128, 128, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())  # enforce Q ≥ 0
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (batch, n_features)

        Returns
        -------
        q_norm : Tensor (batch, 1)
            Predicted normalised heat flux.
        """
        return self.net(x)

    # ── persistence helpers ──────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model state dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: object) -> "TransportMLP":
        """Instantiate model and load state dict from *path*."""
        model = cls(**kwargs)  # type: ignore[arg-type]
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model
