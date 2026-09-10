"""Low-level MLP: SiLU hidden layers and configurable scalar output.

Project training uses a linear output for standardized log diffusivity.
The softplus option remains for standalone, nonnegative regression examples.
Use TransportBundle to load complete project artifacts with their transforms.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class TransportMLP(nn.Module):
    """Feedforward MLP with explicit output-space choice.

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
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        if output_activation == "softplus":
            layers.append(nn.Softplus())
        elif output_activation != "linear":
            raise ValueError("output_activation must be softplus or linear")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (batch, n_features)

        Returns
        -------
        q_norm : Tensor (batch, 1)
            Raw output in the training target space.
        """
        return self.net(x)

    # ── persistence helpers ──────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model state dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: object) -> TransportMLP:
        """Instantiate model and load state dict from *path*."""
        model = cls(**kwargs)  # type: ignore[arg-type]
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model
