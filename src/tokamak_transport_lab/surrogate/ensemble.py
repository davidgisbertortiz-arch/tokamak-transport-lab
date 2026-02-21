"""Deep Ensemble wrapper for epistemic uncertainty quantification.

Wraps *N* independently trained ``torch.nn.Module`` members and provides
aggregate predictions with mean and epistemic standard deviation.

Limitations
-----------
- Epistemic uncertainty only (no aleatoric / heteroscedastic head).
- Members must share the same architecture; only random init and
  data-shuffling provide diversity.
- Population std (``correction=0``) is used so a single-member
  ensemble returns zero uncertainty instead of NaN.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class DeepEnsemble:
    """Ensemble of *N* identical-architecture neural network members.

    Parameters
    ----------
    members : list[nn.Module]
        Pre-trained neural network members (already on the same device).
    """

    def __init__(self, members: list[nn.Module]) -> None:
        if not members:
            raise ValueError("Ensemble requires at least one member.")
        self.members = members
        for m in self.members:
            m.eval()

    # ── forward helpers ──────────────────────────────────────────

    @torch.no_grad()
    def predict_members(self, x: torch.Tensor) -> torch.Tensor:
        """Return stacked predictions from every member.

        Parameters
        ----------
        x : Tensor (batch, n_features)

        Returns
        -------
        preds : Tensor (n_members, batch, out_dim)
        """
        return torch.stack([m(x) for m in self.members], dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Aggregate prediction with epistemic uncertainty.

        Parameters
        ----------
        x : Tensor (batch, n_features)

        Returns
        -------
        result : dict
            ``"mean"``           — (batch, out_dim) ensemble mean.
            ``"epistemic_std"``  — (batch, out_dim) std across member means.
            ``"members"``        — (n_members, batch, out_dim) raw predictions.
        """
        preds = self.predict_members(x)  # (M, B, D)
        return {
            "mean": preds.mean(dim=0),
            "epistemic_std": preds.std(dim=0, correction=0),
            "members": preds,
        }

    # ── persistence ──────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Save each member's state_dict to *directory*/member_<i>.pt."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save(m.state_dict(), directory / f"member_{i}.pt")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        model_factory: type[nn.Module],
        n_members: int,
        **factory_kwargs: object,
    ) -> DeepEnsemble:
        """Load an ensemble from saved member checkpoints.

        Parameters
        ----------
        directory : path
            Folder containing ``member_0.pt`` ... ``member_{n-1}.pt``.
        model_factory : type
            Callable (class) returning an ``nn.Module``.
        n_members : int
            Number of members to load.
        **factory_kwargs
            Forwarded to *model_factory* constructor.
        """
        directory = Path(directory)
        members: list[nn.Module] = []
        for i in range(n_members):
            model = model_factory(**factory_kwargs)  # type: ignore[arg-type]
            sd = torch.load(
                directory / f"member_{i}.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(sd)
            model.eval()
            members.append(model)
        return cls(members)

    def __len__(self) -> int:
        return len(self.members)
