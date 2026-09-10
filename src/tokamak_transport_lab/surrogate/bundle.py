"""Versioned inference artifact: architecture, transforms, feature order, weights.

The network predicts standardized log(chi). A positive inverse transform and
an exact subcritical floor impose the known structure of this synthetic law.
This is a hybrid surrogate of a specified toy closure, not learned plasma physics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from tokamak_transport_lab.data.generate import FEATURE_NAMES
from tokamak_transport_lab.surrogate.mlp import TransportMLP


class TransportBundle:
    def __init__(self, models, *, x_mean, x_std, y_mean, y_std, bounds, hidden, seeds):
        if not models or len(models) != len(seeds):
            raise ValueError("Require one seed per member")
        self.models = [m.cpu().double().eval() for m in models]
        self.x_mean = np.asarray(x_mean)
        self.x_std = np.asarray(x_std)
        self.y_mean, self.y_std = float(y_mean), float(y_std)
        self.bounds = np.asarray(bounds)
        self.hidden, self.seeds = tuple(hidden), list(seeds)
        if self.x_mean.shape != (5,) or self.x_std.shape != (5,) or np.any(self.x_std <= 0):
            raise ValueError("Invalid feature normalization")
        self.fingerprint = None

    def predict_members(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all() or np.any(x[:, -1] <= 0):
            raise ValueError("Expected finite (N,5) closure inputs with positive chi_neo")
        prediction = np.broadcast_to(x[:, 4], (len(self.models), len(x))).copy()
        active = x[:, 0] > x[:, 2]
        if np.any(active):
            xa = x[active].copy()
            boundary = xa.copy()
            boundary[:, 0] = boundary[:, 2]
            inputs = np.concatenate([xa, boundary])
            xt = torch.as_tensor((inputs - self.x_mean) / self.x_std, dtype=torch.float64)
            with torch.no_grad():
                raw = np.array([m(xt).squeeze(-1).numpy() for m in self.models])
            values = np.exp(raw * self.y_std + self.y_mean)
            count = len(xa)
            # Anchor the learned excess to zero at the critical gradient.
            # This makes the closure continuous, unlike a hard switch between
            # unrelated network predictions and the known subcritical floor.
            prediction[:, active] += np.maximum(values[:, :count] - values[:, count:], 0.0)
        if not np.isfinite(prediction).all():
            raise ValueError("Surrogate produced nonfinite diffusivity")
        return prediction

    def predict_chi(self, x, member=None):
        values = self.predict_members(x)
        return values.mean(axis=0) if member is None else values[member]

    def out_of_domain(self, x):
        x = np.asarray(x)
        return np.any((x < self.bounds[:, 0]) | (x > self.bounds[:, 1]), axis=-1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "target": "log_chi",
            "features": FEATURE_NAMES,
            "hidden": list(self.hidden),
            "seeds": self.seeds,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "bounds": self.bounds.tolist(),
            "members": [m.state_dict() for m in self.models],
            "output_activation": "linear",
            "subcritical_floor": True,
            "inference_transform": "critical_gradient_anchored_v1",
        }
        torch.save(payload, path)
        self.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path):
        torch.set_num_threads(1)
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (
            data.get("schema_version") != 2
            or data.get("features") != FEATURE_NAMES
            or data.get("target") != "log_chi"
            or data.get("inference_transform") != "critical_gradient_anchored_v1"
        ):
            raise ValueError(
                "Incompatible artifact; retrain using scripts.train_surrogate/train_ensemble"
            )
        models = []
        for state in data["members"]:
            model = TransportMLP(
                n_features=5, hidden=tuple(data["hidden"]), output_activation="linear"
            ).double()
            model.load_state_dict(state)
            models.append(model)
        obj = cls(
            models,
            **{
                k: data[k]
                for k in ("x_mean", "x_std", "y_mean", "y_std", "bounds", "hidden", "seeds")
            },
        )
        obj.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        return obj
