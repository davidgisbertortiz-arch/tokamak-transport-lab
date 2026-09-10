"""Deep Ensemble for epistemic uncertainty quantification.

Trains *N* independently initialised ``torch.nn.Module`` members and
provides aggregate predictions with mean and epistemic standard deviation.

Limitations
-----------
- Epistemic uncertainty only (no aleatoric / heteroscedastic head).
- Members share the same architecture; random init + data shuffling
  provide diversity.
- Population std (``correction=0``) is used so a single-member
  ensemble returns zero uncertainty instead of NaN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader


class DeepEnsemble:
    """Ensemble of *N* identical-architecture neural network members.

    Parameters
    ----------
    model_factory : callable
        A callable (class or function) returning a fresh ``nn.Module``.
        Any additional keyword arguments are forwarded to it.
    n_members : int
        Number of ensemble members to create.
    seeds : list[int] | None
        Per-member seeds for weight initialisation.  If *None*, uses
        ``[0, 1, ..., n_members - 1]``.
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, …).
    **factory_kwargs
        Forwarded to *model_factory* when building each member.

    Examples
    --------
    >>> ens = DeepEnsemble(TransportMLP, n_members=5, n_features=6,
    ...                    hidden=(128, 64))
    >>> ens.fit(train_loader, epochs=50)
    >>> result = ens.predict(x_test)
    """

    def __init__(
        self,
        model_factory: Callable[..., nn.Module],
        n_members: int = 5,
        seeds: list[int] | None = None,
        device: str = "cpu",
        **factory_kwargs: Any,
    ) -> None:
        self.model_factory = model_factory
        self.n_members = n_members
        self.device = device
        self.factory_kwargs = factory_kwargs

        if seeds is None:
            seeds = list(range(n_members))
        self.seeds = list(seeds)

        # Build members with diverse random initialisations
        self.members: list[nn.Module] = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            model = model_factory(**factory_kwargs)
            model.to(device)
            model.eval()
            self.members.append(model)

    # ── training ─────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> dict[str, Any]:
        """Train every member independently on *train_loader*.

        Parameters
        ----------
        train_loader : DataLoader
            Yields ``(x_batch, y_batch)`` tensors.
        val_loader : DataLoader | None
            Optional validation loader for monitoring.
        epochs : int
            Training epochs per member.
        lr : float
            AdamW learning rate.
        weight_decay : float
            AdamW weight decay.

        Returns
        -------
        histories : dict
            Per-member training and (optionally) validation loss curves.
        """
        loss_fn = nn.MSELoss()
        histories: dict[str, Any] = {}

        for i, (model, seed) in enumerate(zip(self.members, self.seeds, strict=True)):
            torch.manual_seed(seed)
            optimiser = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=epochs,
            )

            train_losses: list[float] = []
            val_losses: list[float] = []

            for _epoch in range(epochs):
                # ── train ──
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimiser.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimiser.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                scheduler.step()
                train_losses.append(epoch_loss / max(n_batches, 1))

                # ── validate ──
                if val_loader is not None:
                    model.eval()
                    with torch.no_grad():
                        vl = sum(
                            loss_fn(
                                model(xb.to(self.device)),
                                yb.to(self.device),
                            ).item()
                            for xb, yb in val_loader
                        )
                    val_losses.append(vl / max(len(val_loader), 1))

            model.eval()
            histories[f"member_{i}"] = {
                "train_loss": train_losses,
                "val_loss": val_losses or None,
            }

        return histories

    # ── inference ────────────────────────────────────────────────

    @torch.no_grad()
    def predict_members(self, x: torch.Tensor) -> torch.Tensor:
        """Return stacked predictions from every member.

        Returns
        -------
        preds : Tensor (n_members, batch, out_dim)
        """
        x = x.to(self.device)
        for m in self.members:
            m.eval()
        return torch.stack([m(x) for m in self.members], dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Aggregate prediction with epistemic uncertainty.

        Returns
        -------
        result : dict
            ``"mean"``           — (batch, out_dim) ensemble mean.
            ``"epistemic_std"``  — (batch, out_dim) std across members.
            ``"member_means"``   — (n_members, batch, out_dim) raw outputs.
        """
        preds = self.predict_members(x)
        return {
            "mean": preds.mean(dim=0),
            "epistemic_std": preds.std(dim=0, correction=0),
            "member_means": preds,
        }

    # ── persistence ──────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Save member checkpoints and metadata to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save(m.state_dict(), directory / f"member_{i}.pt")
        metadata = {
            "n_members": self.n_members,
            "seeds": self.seeds,
            "device": self.device,
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        model_factory: Callable[..., nn.Module],
        n_members: int | None = None,
        **factory_kwargs: Any,
    ) -> DeepEnsemble:
        """Load an ensemble from saved checkpoints.

        Parameters
        ----------
        directory : path
            Folder containing ``member_0.pt`` … and optionally
            ``metadata.json``.
        model_factory : callable
            Same factory used to create the ensemble.
        n_members : int | None
            Number of members.  Read from metadata.json if omitted.
        **factory_kwargs
            Forwarded to *model_factory*.
        """
        directory = Path(directory)
        meta_path = directory / "metadata.json"
        seeds: list[int] | None = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_members = n_members or meta["n_members"]
            seeds = meta.get("seeds")
        if n_members is None:
            msg = "n_members must be specified when metadata.json is absent."
            raise ValueError(msg)

        ens = cls(
            model_factory,
            n_members=n_members,
            seeds=seeds,
            **factory_kwargs,
        )
        for i, model in enumerate(ens.members):
            sd = torch.load(
                directory / f"member_{i}.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(sd)
            model.eval()
        return ens

    def __len__(self) -> int:
        return len(self.members)
